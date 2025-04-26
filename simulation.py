import flwr as fl
import client
import os
from server import get_server_strategy as get_fedavg_strategy
from FedAGRU_server import get_server_strategy as get_fedagru_strategy

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CLIENTS = int(os.getenv("CLIENTS", 3))
# Get partition type from command-line argument or default to "iid"
PARTITION_TYPE = os.getenv("PARTITION_TYPE", "iid")
assert PARTITION_TYPE in ["iid", "quantity_skew", "label_skew"], "Invalid partition type"

# Get strategy type from environment variable or default to "fedavg"
STRATEGY_TYPE = os.getenv("STRATEGY_TYPE", "fedavg")
assert STRATEGY_TYPE in ["fedavg", "fedagru"], "Invalid strategy type"

def create_client(cid):
    return client.Client(cid, NUM_CLIENTS, partition_type=PARTITION_TYPE)

if __name__ == "__main__":
    # Select strategy based on STRATEGY_TYPE
    if STRATEGY_TYPE == "fedavg":
        strategy = get_fedavg_strategy()
    else:  # fedagru
        strategy = get_fedagru_strategy(NUM_CLIENTS)

    history = fl.simulation.start_simulation(
        client_fn=create_client,
        num_clients=NUM_CLIENTS,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")