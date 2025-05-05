#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Function to run a test and save output
run_test() {
    local test_name=$1
    local dataset=$2
    local strategy=$3
    local partition=$4
    local clients=$5
    
    echo "Running test: $test_name"
    echo "Configuration:"
    echo "- Dataset: $dataset"
    echo "- Strategy: $strategy"
    echo "- Partition: $partition"
    echo "- Clients: $clients"
    
    # Run the simulation and save output
    DATASET_TYPE=$dataset \
    STRATEGY_TYPE=$strategy \
    PARTITION_TYPE=$partition \
    CLIENTS=$clients \
    python3 simulation.py > "results/${test_name}.log" 2>&1
    
    echo "Test completed. Results saved to results/${test_name}.log"
    echo "----------------------------------------"
}

# Test 1: Basic UNSW with FedAvg
run_test "test1_unsw_fedavg_iid" "unsw" "fedavg" "iid" "3"

# Test 2: Basic CIC with FedAvg
run_test "test2_cic_fedavg_iid" "cic" "fedavg" "iid" "3"

# Test 3: UNSW with FedAGRU
run_test "test3_unsw_fedagru_iid" "unsw" "fedagru" "iid" "3"

# Test 4: CIC with FedAGRU
run_test "test4_cic_fedagru_iid" "cic" "fedagru" "iid" "3"

# Test 5: UNSW with quantity skew
run_test "test5_unsw_fedavg_quantity" "unsw" "fedavg" "quantity_skew" "5"

# Test 6: CIC with quantity skew
run_test "test6_cic_fedavg_quantity" "cic" "fedavg" "quantity_skew" "5"

# Test 7: UNSW with label skew
run_test "test7_unsw_fedagru_label" "unsw" "fedagru" "label_skew" "5"

# Test 8: CIC with label skew
run_test "test8_cic_fedagru_label" "cic" "fedagru" "label_skew" "5"

# Test 9: UNSW with more clients
run_test "test9_unsw_fedavg_many_clients" "unsw" "fedavg" "iid" "10"

# Test 10: CIC with more clients
run_test "test10_cic_fedagru_many_clients" "cic" "fedagru" "iid" "10"

# Test 11: UNSW with quantity skew and more clients
run_test "test11_unsw_fedavg_quantity_many" "unsw" "fedavg" "quantity_skew" "10"

# Test 12: CIC with quantity skew and more clients  
run_test "test12_cic_fedavg_quantity_many" "cic" "fedavg" "quantity_skew" "10"

# Test 13: UNSW with label skew and FedAGRU with more clients
run_test "test13_unsw_fedagru_label_many" "unsw" "fedagru" "label_skew" "10"

# Test 14: CIC with label skew and FedAGRU with more clients
run_test "test14_cic_fedagru_label_many" "cic" "fedagru" "label_skew" "10"

# Test 15: UNSW with very large number of clients
run_test "test15_unsw_fedavg_massive" "unsw" "fedavg" "iid" "15"

# Test 16: CIC with very large number of clients
run_test "test16_cic_fedagru_massive" "cic" "fedagru" "iid" "15"

echo "All tests completed. Results are in the 'results' directory." 