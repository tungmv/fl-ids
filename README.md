# Federated Learning for Intrusion Detection System

## Overview
This project utilizes the power of Federated Learning to create an Intrusion Detection System. It employs the Flower framework ([GitHub - adap/flower](https://github.com/adap/flower)) for federated learning and the [UNSW_NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for intrusion detection. The core of this system is a Neural Network (NN) model.

## Setup & Installation
To get the project up and running, follow these steps:
1. Download the UNSW_NB15 dataset.
2. Place the `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` files in the project's data folder.
3. Optionally, download the CIC-IDS2023 dataset if you want to use it.
   Place the `cic23_train.csv` and `cic23_test.csv` files in the data folder.

## Data Partitioning
The system offers two ways to partition data among clients:

1. **On-the-fly partitioning**: Data is partitioned when clients are initialized (default)
2. **Pre-partitioned data**: Data is partitioned once and saved to disk for reuse

### Pre-partitioning Options
Pre-partitioning data can speed up experiments, especially with large datasets or many clients.

#### Generate Partitions
Use the dedicated script:
```shell
python generate_partitions.py --dataset unsw --partition-type iid --num-clients 3 --output-dir data/partitions
```

Or use the simulation script with the `--generate-only` flag:
```shell
GENERATE_PARTITIONS=1 python simulation.py --generate-only
```

#### Use Pre-partitioned Data
Set the `USE_PREPARTITIONED` environment variable:
```shell
USE_PREPARTITIONED=1 python simulation.py
```

## Simulation
There are 3 options to execute simulation:

1. Manually
    - To initiate the server, use the command:
        ```shell
        python server.py
        ```
    - To launch the clients, execute the following command for each client:
        ```shell
        python client.py
        ```
    **Note:** To accurately simulate the project, at least three clients are needed to satisfy the `min_fit_clients`, `min_evaluate_clients`, and `min_available_clients` configuration.
    
    **Note:** The server uses port 8080 as default. If you want to run it on a different port, you may need to change it in both `client.py` and `server.py`.

2. Using the `simulation.py` script
    - To run the simulation, use the command:
        ```shell
        python simulation.py
        ```
    - Additional environment variables:
        ```shell
        # Control number of clients
        CLIENTS=5 python simulation.py
        
        # Choose dataset (unsw or cic)
        DATASET_TYPE=cic python simulation.py
        
        # Choose partition strategy (iid, quantity_skew, label_skew)
        PARTITION_TYPE=label_skew python simulation.py
        
        # Choose FL strategy (fedavg or fedagru)
        STRATEGY_TYPE=fedagru python simulation.py
        
        # Use pre-partitioned data
        USE_PREPARTITIONED=1 python simulation.py
        ```

3. Containerized simulation by running the Docker compose command
    - To run the simulation, use the command:
        ```shell
        docker-compose up --build
        ```

Also you can visualize the model's architecture by generating a plot

- To visualize the model's structure, use:
    ```shell
    python utils/plot.py
    ```

## Future Enhancements
- Add more federated learning strategies
- Implement additional IDS models and datasets