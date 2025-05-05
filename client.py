import os
import flwr as fl
import utils.data_loader as data_loader
import utils.model_loader as model_loader
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get dataset type from environment variable or default to "unsw"
DATASET_TYPE = os.getenv("DATASET_TYPE", "unsw")
assert DATASET_TYPE in ["unsw", "cic"], "Invalid dataset type"

class PartitioningStrategy:
    # Cache for distributions to ensure consistency across clients
    _quantity_distributions = {}  # Experiment seed -> proportions array
    _label_distributions = {}     # Experiment seed -> client -> label distributions
    
    @staticmethod
    def _get_experiment_seed():
        """Get a consistent seed for the experiment based on environment variables"""
        # Use experiment ID or default to a fixed value for reproducibility
        return int(os.getenv("EXPERIMENT_SEED", "42"))
    
    @staticmethod
    def iid_partition(X, Y, num_clients: int, client_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """IID partitioning: Uniform random split among clients after shuffling"""
        # Create a deterministic random state based on experiment seed
        seed = PartitioningStrategy._get_experiment_seed()
        rng = np.random.RandomState(seed)
        
        # Generate shuffled indices
        n_samples = len(X)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        # Divide shuffled data among clients
        samples_per_client = n_samples // num_clients
        start_idx = client_idx * samples_per_client
        end_idx = start_idx + samples_per_client if client_idx < num_clients - 1 else n_samples
        
        selected_indices = indices[start_idx:end_idx]
        return X[selected_indices], Y[selected_indices]

    @staticmethod
    def quantity_skew_partition(X, Y, num_clients: int, client_idx: int, 
                              min_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Non-IID partitioning with quantity skew: Different amounts of data per client"""
        n_samples = len(X)
        seed = PartitioningStrategy._get_experiment_seed()
        
        # Use cached distribution or create a new one
        if seed not in PartitioningStrategy._quantity_distributions:
            # Set seed for reproducibility
            np.random.seed(seed)
            # Create dirichlet distribution for quantity skew
            proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
            # Ensure minimum samples per client if specified
            if min_samples:
                proportions = np.clip(proportions, min_samples/n_samples, 1.0)
                proportions = proportions / proportions.sum()
            
            PartitioningStrategy._quantity_distributions[seed] = proportions
            
        proportions = PartitioningStrategy._quantity_distributions[seed]
        
        # Create a shuffled dataset first for true randomness
        rng = np.random.RandomState(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        # Calculate cumulative sample indices
        cumsum = np.round(np.cumsum(proportions) * n_samples).astype(int)
        cumsum[-1] = n_samples  # Ensure we use all samples
        
        # Get this client's partition
        start_idx = 0 if client_idx == 0 else cumsum[client_idx-1]
        end_idx = cumsum[client_idx]
        
        selected_indices = indices[start_idx:end_idx]
        return X[selected_indices], Y[selected_indices]

    @staticmethod
    def label_skew_partition(X, Y, num_clients: int, client_idx: int, 
                           concentration: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Non-IID partitioning with label skew: Different label distributions per client"""
        seed = PartitioningStrategy._get_experiment_seed()
        unique_labels = np.unique(Y)
        
        # Initialize distributions cache for this experiment if needed
        if seed not in PartitioningStrategy._label_distributions:
            PartitioningStrategy._label_distributions[seed] = {}
            
            # Set seed for reproducibility
            np.random.seed(seed)
            
            # Create label-client assignment matrix (rows=clients, cols=labels)
            # Each row is a distribution over labels for that client
            all_distributions = np.random.dirichlet(
                np.repeat(concentration, len(unique_labels)), 
                size=num_clients
            )
            
            # Store all client distributions
            for idx in range(num_clients):
                PartitioningStrategy._label_distributions[seed][idx] = all_distributions[idx]
        
        # Get this client's label distribution
        label_distribution = PartitioningStrategy._label_distributions[seed][client_idx]
        
        # For consistency in sampling, use a deterministic RNG
        rng = np.random.RandomState(seed + client_idx)
        
        # Organize data by label
        label_to_indices = {label: np.where(Y == label)[0] for label in unique_labels}
        
        # Create a client-specific partition for each label
        # To avoid overlaps, we'll divide indices for each label among clients
        selected_indices = []
        for label_idx, label in enumerate(unique_labels):
            label_indices = label_to_indices[label]
            # Shuffle indices for this label
            rng.shuffle(label_indices)
            
            # Determine how many samples for this client based on distribution
            proportion_for_client = label_distribution[label_idx]
            n_samples_total = len(label_indices)
            client_samples = int(proportion_for_client * n_samples_total / num_clients)
            
            # Take a unique slice for this client
            start = client_idx * client_samples
            end = min(start + client_samples, n_samples_total)
            client_indices = label_indices[start:end]
            selected_indices.extend(client_indices)
        
        selected_indices = np.array(selected_indices)
        
        # Shuffle final selection for good measure
        rng.shuffle(selected_indices)
        
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
        # Load full dataset based on DATASET_TYPE
        if DATASET_TYPE == "unsw":
            X_train, Y_train, X_test, Y_test = data_loader.get_data()
        else:  # cic
            X_train, Y_train, X_test, Y_test = data_loader.get_data_cic()
            
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
              f"and {len(self.X_test)} test samples using {DATASET_TYPE.upper()} dataset")
        
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