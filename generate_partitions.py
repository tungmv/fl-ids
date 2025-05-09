#!/usr/bin/env python
"""
Generate and save pre-partitioned data for Federated Learning experiments.
This script creates partitioned datasets according to specified parameters
and saves them for later use by clients.
"""
import os
import argparse
import client

def main():
    parser = argparse.ArgumentParser(description="Generate pre-partitioned data for FL experiments")
    parser.add_argument("--dataset", type=str, choices=["unsw", "cic"], default="unsw",
                        help="Dataset to use (unsw or cic)")
    parser.add_argument("--partition-type", type=str, choices=["iid", "quantity_skew", "label_skew"], 
                        default="iid", help="Type of partitioning")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--output-dir", type=str, default="data/partitions", 
                        help="Output directory for partitions")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["DATASET_TYPE"] = args.dataset
    os.environ["PARTITION_DIR"] = args.output_dir
    
    # Print generation info
    print(f"Generating partitions with the following parameters:")
    print(f"- Dataset: {args.dataset.upper()}")
    print(f"- Partition type: {args.partition_type}")
    print(f"- Number of clients: {args.num_clients}")
    print(f"- Output directory: {args.output_dir}")
    
    # Generate and save partitions
    client.save_partitions(args.partition_type, args.num_clients, dataset_type=args.dataset)
    
    print("Partitioning complete!")

if __name__ == "__main__":
    main()