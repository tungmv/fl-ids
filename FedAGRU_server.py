import flwr as fl
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from flwr.common import (
    EvaluateRes,
    FitRes, 
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / total_examples for k, v in federated_metrics.items()}

class FedAGRU(FedAvg):
    """Federated Attentive Gated Recurrent Unit (FedAGRU) strategy.
    
    This is a custom implementation of FedAGRU that extends FedAvg with an attention
    mechanism to assign importance weights to client updates.
    """
    
    def __init__(
        self,
        *,
        attention_decay: float = 0.9,  # Decay factor for attention weights
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs,
        )
        self.attention_decay = attention_decay
        # Initialize client attention weights (empty dict to start)
        self.client_attention: Dict[str, float] = {}
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates using attention weights."""
        if not results:
            return None, {}
        
        # Update attention weights based on client performance
        for client, fit_res in results:
            client_id = client.cid
            # Use accuracy or loss as performance metric
            client_loss = fit_res.metrics.get("loss", 0.0)
            
            # Update attention weight with decay factor
            # Higher loss = lower attention weight
            if client_id in self.client_attention:
                prev_weight = self.client_attention[client_id]
                # Normalize loss to be between 0 and 1 (approximation)
                normalized_performance = 1.0 / (1.0 + client_loss)
                # Apply decay and update
                self.client_attention[client_id] = (
                    self.attention_decay * prev_weight +
                    (1 - self.attention_decay) * normalized_performance
                )
            else:
                # Initialize with normalized performance
                normalized_performance = 1.0 / (1.0 + client_loss)
                self.client_attention[client_id] = normalized_performance
        
        # Get total attention weights
        total_attention = sum(self.client_attention.values())
        
        # Convert results to list of weights and num_examples
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples * (self.client_attention.get(client.cid, 1.0) / max(total_attention, 1e-10))
            )
            for client, fit_res in results
        ]
        
        # Aggregate parameters weighted by attention-adjusted num_examples
        parameters_aggregated = self.aggregate_parameters(weights_results)
        
        # Convert parameters back to required format
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        
        # Also return statistics for clients
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(metrics)
        
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_parameters(self, results):
        """Aggregate parameters based on weighted results."""
        # Extract weights and their corresponding importance
        weights = [parameters for parameters, _ in results]
        attention_weights = np.array([importance for _, importance in results])
        
        # Normalize weights
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Create a list of aggregated weights
        aggregated_weights = []
        for i in range(len(weights[0])):
            # Extract the i-th weight from each client
            layer_updates = [w[i] for w in weights]
            # Apply weighted aggregation
            weighted_update = np.sum(
                [update * weight for update, weight in zip(layer_updates, attention_weights)],
                axis=0,
            )
            aggregated_weights.append(weighted_update)
        
        return aggregated_weights

def get_server_strategy(num_clients=10):
    return FedAGRU(
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
if __name__ == "__main__":
    import sys
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=get_server_strategy(num_clients),
        config=fl.server.ServerConfig(num_rounds=10),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
