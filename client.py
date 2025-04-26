import os
import flwr as fl
import utils.data_loader as data_loader
import utils.model_loader as model_loader
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PartitioningStrategy:
    @staticmethod
    def iid_partition(X, Y, num_clients: int, client_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """IID partitioning: Uniform random split among clients"""
        n_samples = len(X)
        samples_per_client = n_samples // num_clients
        start_idx = client_idx * samples_per_client
        end_idx = start_idx + samples_per_client
        return X[start_idx:end_idx], Y[start_idx:end_idx]

    @staticmethod
    def quantity_skew_partition(X, Y, num_clients: int, client_idx: int, 
                              min_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Non-IID partitioning with quantity skew: Different amounts of data per client"""
        n_samples = len(X)
        # Create dirichlet distribution for quantity skew
        proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
        # Ensure minimum samples per client if specified
        if min_samples:
            proportions = np.clip(proportions, min_samples/n_samples, 1.0)
            proportions = proportions / proportions.sum()
        
        # Calculate cumulative sample indices
        cumsum = np.round(np.cumsum(proportions) * n_samples).astype(int)
        start_idx = 0 if client_idx == 0 else cumsum[client_idx-1]
        end_idx = cumsum[client_idx]
        return X[start_idx:end_idx], Y[start_idx:end_idx]

    @staticmethod
    def label_skew_partition(X, Y, num_clients: int, client_idx: int, 
                           concentration: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Non-IID partitioning with label skew: Different label distributions per client"""
        unique_labels = np.unique(Y)
        # Create label distribution for each client using Dirichlet distribution
        label_distribution = np.random.dirichlet(np.repeat(concentration, len(unique_labels)))
        
        # Select samples based on label distribution
        selected_indices = []
        for label_idx, label in enumerate(unique_labels):
            label_indices = np.where(Y == label)[0]
            n_samples = int(len(label_indices) * label_distribution[label_idx])
            selected_indices.extend(np.random.choice(label_indices, n_samples, replace=False))
        
        selected_indices = np.array(selected_indices)
        return X[selected_indices], Y[selected_indices]

class Client(fl.client.NumPyClient):
    def __init__(self, cid: str, num_clients: int, partition_type: str = "iid"):
        """
        Initialize client with specific partitioning strategy
        Args:
            cid: Client ID
            num_clients: Total number of clients
            partition_type: One of ["iid", "quantity_skew", "label_skew"]
        """
        # Load full dataset
        X_train, Y_train, X_test, Y_test = data_loader.get_data()
        client_idx = int(cid)

        # Select partitioning strategy
        if partition_type == "iid":
            self.X_train, self.Y_train = PartitioningStrategy.iid_partition(
                X_train, Y_train, num_clients, client_idx)
            self.X_test, self.Y_test = PartitioningStrategy.iid_partition(
                X_test, Y_test, num_clients, client_idx)
        
        elif partition_type == "quantity_skew":
            self.X_train, self.Y_train = PartitioningStrategy.quantity_skew_partition(
                X_train, Y_train, num_clients, client_idx, min_samples=100)
            self.X_test, self.Y_test = PartitioningStrategy.quantity_skew_partition(
                X_test, Y_test, num_clients, client_idx, min_samples=50)
        
        elif partition_type == "label_skew":
            self.X_train, self.Y_train = PartitioningStrategy.label_skew_partition(
                X_train, Y_train, num_clients, client_idx)
            self.X_test, self.Y_test = PartitioningStrategy.label_skew_partition(
                X_test, Y_test, num_clients, client_idx)
        
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")

        print(f"Client {cid} initialized with {len(self.X_train)} training samples "
              f"and {len(self.X_test)} test samples")
        
        # Initialize the model
        self.model = model_loader.get_model(self.X_train.shape[1:])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, _):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.Y_train, epochs=1, batch_size=64)
        return self.model.get_weights(), len(self.X_train), {k: v[-1] for k, v in history.history.items()}

    def evaluate(self, parameters, _):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_numpy_client(server_address=server_address, client=Client())