import flwr as fl
import client
import os
from server import get_server_strategy 

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CLIENTS = int(os.getenv("CLIENTS", 3))
# Get partition type from command-line argument or default to "iid"
PARTITION_TYPE = os.getenv("PARTITION_TYPE", "iid")
assert PARTITION_TYPE in ["iid", "quantity_skew", "label_skew"], "Invalid partition type"

def create_client(cid):
    return client.Client(cid, NUM_CLIENTS, partition_type=PARTITION_TYPE)

if __name__ == "__main__":
    history = fl.simulation.start_simulation(
        client_fn=create_client,
        num_clients=NUM_CLIENTS,
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
