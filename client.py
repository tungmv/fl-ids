import os
import flwr as fl
import utils.data_loader as data_loader
import utils.model_loader as model_loader
import numpy as np

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Client(fl.client.NumPyClient):
    def __init__(self, cid: str, num_clients: int):
        # Load full dataset
        X_train, Y_train, X_test, Y_test = data_loader.get_data()
        
        # Calculate partition indices
        train_size = len(X_train)
        test_size = len(X_test)
        train_items_per_client = train_size // num_clients
        test_items_per_client = test_size // num_clients
        
        # Get client's partition
        client_idx = int(cid)
        train_start = client_idx * train_items_per_client
        train_end = train_start + train_items_per_client
        test_start = client_idx * test_items_per_client
        test_end = test_start + test_items_per_client
        
        # Partition the data for this client
        self.X_train = X_train[train_start:train_end]
        self.Y_train = Y_train[train_start:train_end]
        self.X_test = X_test[test_start:test_end]
        self.Y_test = Y_test[test_start:test_end]
        
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