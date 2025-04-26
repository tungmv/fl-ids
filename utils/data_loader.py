import pandas as pd
import numpy as np
from sklearn import preprocessing

def get_data_cic():
    """Load and preprocess CIC-IDS2017 dataset."""
    train_data = pd.read_csv("data/cic23_train.csv")
    test_data = pd.read_csv("data/cic23_test.csv")

    # Drop unnecessary columns if they exist
    columns_to_drop = ['id', 'attack_cat', 'attack_type']
    for col in columns_to_drop:
        if col in train_data.columns:
            train_data = train_data.drop(columns=[col])
        if col in test_data.columns:
            test_data = test_data.drop(columns=[col])

    # Preprocess data
    train_data = _preprocess_data(train_data)
    test_data = _preprocess_data(test_data)

    # Separate features and labels
    X_train, Y_train = _separate_features_and_labels(train_data)
    X_test, Y_test = _separate_features_and_labels(test_data)

    return X_train, Y_train, X_test, Y_test

def get_data():
    """Load and preprocess UNSW-NB15 dataset."""
    train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
    test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")

    # Drop unnecessary columns if they exist
    columns_to_drop = ['id', 'attack_cat']
    for col in columns_to_drop:
        if col in train_data.columns:
            train_data = train_data.drop(columns=[col])
        if col in test_data.columns:
            test_data = test_data.drop(columns=[col])

    # Preprocess data
    train_data = _preprocess_data(train_data)
    test_data = _preprocess_data(test_data)

    # Separate features and labels
    X_train, Y_train = _separate_features_and_labels(train_data)
    X_test, Y_test = _separate_features_and_labels(test_data)

    return X_train, Y_train, X_test, Y_test

def _preprocess_data(data):
    """Preprocess the data by encoding categorical features and scaling numerical features."""
    # Encode categorical features
    for column in data.columns:
        if data[column].dtype == 'object':
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))

    # Scale numerical features
    min_max_scaler = preprocessing.MinMaxScaler()
    for column in data.columns:
        if column != 'label':  # Don't scale the label column
            data[column] = min_max_scaler.fit_transform(data[column].values.reshape(-1,1))
    
    return data

def _separate_features_and_labels(data):
    """Separate features and labels from the dataset."""
    Y = data.label.to_numpy()
    X = data.drop(columns="label").to_numpy()
    return X, Y