#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results/configs results/logs

# Initialize summary file
SUMMARY_FILE="results/test_summary.csv"
echo "Test ID,Dataset,Strategy,Partition,Clients,Status,Runtime(s),Accuracy,Loss" > $SUMMARY_FILE

# Function to run a test and save output
run_test() {
    local test_name=$1
    local dataset=$2
    local strategy=$3
    local partition=$4
    local clients=$5

    echo "Running test: $test_name"
    echo "Configuration: $dataset $strategy ($partition partition, $clients clients)"

    # Record start time and run the simulation
    start_time=$(date +%s)
    echo "Running: DATASET_TYPE=$dataset STRATEGY_TYPE=$strategy PARTITION_TYPE=$partition CLIENTS=$clients python3 simulation.py"
    DATASET_TYPE=$dataset STRATEGY_TYPE=$strategy PARTITION_TYPE=$partition CLIENTS=$clients \
    timeout 3600 python3 simulation.py > "results/logs/${test_name}.log" 2>&1
    status=$?
    runtime=$(($(date +%s) - start_time))

    # Extract metrics from log
    accuracy=$(grep -o "After .* rounds of training the accuracy is [0-9.]*%" "results/logs/${test_name}.log" | sed -E 's/.*accuracy is ([0-9.]*)%.*/\1/' || echo "N/A")
    loss=$(grep -A 10 "'loss'" "results/logs/${test_name}.log" | grep -o "(3, [0-9.]*)" | head -1 | sed -E 's/\(3, ([0-9.]*)\)/\1/' || echo "N/A")

    # Report results
    if [ $status -eq 0 ]; then
        echo "✓ Success (${runtime}s): Accuracy=${accuracy}%, Loss=${loss}"
        status_text="SUCCESS"
    else
        echo "✗ Failed (${runtime}s)"
        status_text="FAILED"
    fi

    # Add to summary
    echo "${test_name#test},${dataset},${strategy},${partition},${clients},${status_text},${runtime},${accuracy},${loss}" >> $SUMMARY_FILE
    echo "----------------------------------------"
}
# Run tests for all configurations
echo "Starting test batch..."

# Create array of parameters
clients_arr=(5 10 15 20 30)
partition_arr=("iid" "label_skew" "quantity_skew")
strategy_arr=("fedavg" "fedagru")
#dataset_arr=("cic" "unsw")
dataset_arr=("unsw")

# Run all combinations
for dataset in "${dataset_arr[@]}"; do
    for strategy in "${strategy_arr[@]}"; do
        for partition in "${partition_arr[@]}"; do
            for clients in "${clients_arr[@]}"; do
                test_id="${dataset}_${strategy}_${partition}_${clients}"
                run_test "$test_id" "$dataset" "$strategy" "$partition" "$clients"
            done
        done
    done
done

# Generate report
echo "Generating summary report..."
{
    echo "===== FL-IDS Test Summary ====="
    echo "Total tests: $(grep -c "," $SUMMARY_FILE | awk '{print $1-1}')"
    echo "Successful tests: $(grep -c "SUCCESS" $SUMMARY_FILE)"
    echo "Failed tests: $(grep -c "FAILED" $SUMMARY_FILE)"
    echo ""

    # Show top performers by accuracy
    echo "Best performing configurations by accuracy:"
    grep "SUCCESS" $SUMMARY_FILE | sort -t ',' -k8 -nr | head -5 |
      awk -F ',' '{printf "- %s with %s (%s partition, %s clients): Accuracy=%s%%, Loss=%s\n",
                          $2, $3, $4, $5, $8, $9}'

    echo ""
    echo "Best performing configurations by lowest loss:"
    grep "SUCCESS" $SUMMARY_FILE | sort -t ',' -k9 -n | head -5 |
      awk -F ',' '{printf "- %s with %s (%s partition, %s clients): Accuracy=%s%%, Loss=%s\n",
                          $2, $3, $4, $5, $8, $9}'
} > "results/summary_report.txt"

echo "All tests completed. Results in 'results' directory."
echo "Summary report available at results/summary_report.txt"
