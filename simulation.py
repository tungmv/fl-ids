import flwr as fl
import client
import os
import argparse
from server import get_server_strategy as get_fedavg_strategy
from FedAGRU_server import get_server_strategy as get_fedagru_strategy

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get number of clients from environment variable or default to 3
NUM_CLIENTS = int(os.getenv("CLIENTS", 3))

# Get partition type from environment variable or default to "iid"
PARTITION_TYPE = os.getenv("PARTITION_TYPE", "iid")
assert PARTITION_TYPE in ["iid", "quantity_skew", "label_skew"], "Invalid partition type"

# Get strategy type from environment variable or default to "fedavg"
STRATEGY_TYPE = os.getenv("STRATEGY_TYPE", "fedavg")
assert STRATEGY_TYPE in ["fedavg", "fedagru"], "Invalid strategy type"

# Get dataset type from environment variable or default to "unsw"
DATASET_TYPE = os.getenv("DATASET_TYPE", "unsw")
assert DATASET_TYPE in ["unsw", "cic"], "Invalid dataset type"

# Check if using pre-partitioned data
USE_PREPARTITIONED = os.getenv("USE_PREPARTITIONED", "0") == "1"

# Check if we should generate partitions
GENERATE_PARTITIONS = os.getenv("GENERATE_PARTITIONS", "0") == "1"

# Directory for pre-partitioned data
PARTITION_DIR = os.getenv("PARTITION_DIR", "data/partitions")

def create_client(cid):
    return client.Client(cid, NUM_CLIENTS, partition_type=PARTITION_TYPE)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Federated Learning simulation")
    parser.add_argument("--generate-only", action="store_true", help="Only generate partitions without running simulation")
    args = parser.parse_args()
    
    # Handle partition generation
    if GENERATE_PARTITIONS or args.generate_only:
        print(f"Pre-generating partitions:")
        print(f"- Dataset: {DATASET_TYPE.upper()}")
        print(f"- Partition: {PARTITION_TYPE}")
        print(f"- Number of clients: {NUM_CLIENTS}")
        print(f"- Output directory: {PARTITION_DIR}")
        
        # Set environment variable for later use by clients
        os.environ["USE_PREPARTITIONED"] = "1"
        
        # Call the pre-partition function
        client.save_partitions(PARTITION_TYPE, NUM_CLIENTS, dataset_type=DATASET_TYPE)
        
        # Exit if only generating partitions
        if args.generate_only:
            print("Partition generation completed. Exiting without running simulation.")
            exit(0)
    
    # Print simulation info
    print(f"Starting simulation with:")
    print(f"- Strategy: {STRATEGY_TYPE.upper()}")
    print(f"- Dataset: {DATASET_TYPE.upper()}")
    print(f"- Partition: {PARTITION_TYPE}")
    print(f"- Number of clients: {NUM_CLIENTS}")
    print(f"- Using pre-partitioned data: {USE_PREPARTITIONED}")

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