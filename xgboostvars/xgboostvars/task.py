"""FedSec: A Flower / TensorFlow app."""

import os
import xgboost as xgb
import numpy as np
import pandas as pd
from keras import layers
from flwr_datasets.partitioner import IidPartitioner
from datasets import load_dataset
from glob import glob
from datasets import Dataset
import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class XGBoostWrapper:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.is_fitted = False

    def get_weights(self):
        if not self.is_fitted:
            return [np.array([], dtype=np.uint8)]  # Empty placeholder
        # Save booster as bytes and convert to a NumPy array
        booster_bytes = self.model.get_booster().save_raw()
        return [np.frombuffer(booster_bytes, dtype=np.uint8)]

    def set_weights(self, weights):
        if len(weights[0]) == 0:
            return  # No weights to set
        booster_bytes = weights[0].tobytes()
        self.model = xgb.XGBClassifier()
        self.model.get_booster().load_raw(booster_bytes)
        self.is_fitted = True

    def fit(self, X, y, **kwargs):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        self.model.fit(X, y)
        self.is_fitted = True
        # Return dummy history compatible with Keras-style output
        accuracy = self.model.score(X, y)
        return type('History', (), {'history': {'loss': [0], 'accuracy': [accuracy]}})()

    def evaluate(self, X, y, **kwargs):
        if not self.is_fitted:
            return 0.0, 0.0
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        accuracy = self.model.score(X, y)
        return 0.0, accuracy  # Loss set to 0 for compatibility
    

def load_model(input_shape, model_type):
    if model_type == "xgboost":
        model = XGBoostWrapper()
        return model
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.BatchNormalization(),  # Add normalization
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    

    # Use a more sophisticated optimizer configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


fds = None  # Cache FederatedDataset

# Mapping for the "Stage" column
stage_mapping = {
    'BENIGN': 0,        # Benign
    'Benign': 0,        # Benign
    'Data Exfiltration': 1,        # Malicious
    'Establish Foothold': 1,       # Malicious
    'Lateral Movement': 1,         # Malicious
    'Reconnaissance': 1            # Malicious
}

def preprocess_csv_files(data_files):
    """Read and concatenate CSV files, apply the stage mapping, and drop the Activity column."""
    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        # Map the 'Stage' column using the provided mapping
        df['Stage'] = df['Stage'].map(stage_mapping)
        # Drop the 'Activity' column if it exists
        
        df = df.drop(columns=['Activity'])
        df = df.drop(columns=['Flow ID'])
        df = df.drop(columns=['Src IP'])
        df = df.drop(columns=['Dst IP'])
        df = df.drop(columns=['Timestamp'])
            
        dfs.append(df)
    # Concatenate all DataFrames into one
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def load_data(partition_id, num_partitions):
    """Load, preprocess, and partition the dataset."""
    data_directory = "C:\\Users\\brahim\\Desktop\\PFE2025\\Dataset\\csv\\"
    file_pattern = os.path.join(data_directory, "*.csv")
    data_files = glob(file_pattern)
    
    # Preprocess CSV files: read, concatenate, apply mapping, and drop the Activity column.
    df = preprocess_csv_files(data_files)
    
    # Drop rows where 'Stage' is missing so that features and labels have the same cardinality.
    df = df.dropna(subset=['Stage']).reset_index(drop=True)
    
    # Build the dataset:
    # All columns except 'Stage' are features.
    features = df.drop(columns=['Stage'])
    # The 'Stage' column is the label.
    target = df['Stage']
    
    # Convert to NumPy arrays with explicit dtypes.
    features_array = features.values.astype(np.float32)  # Convert features to float32.
    labels_array = target.values.astype(np.int8)          # Convert labels to int64.
    
    dataset_dict = {
        'features': features_array,
        'label': labels_array
    }
    
    # Create a Hugging Face Dataset from the dictionary.
    dataset = Dataset.from_dict(dataset_dict)
    
    # Create an IID partitioner and then a FederatedDataset.
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
        
    # Split the partition into 80% training and 20% testing.
    data = partition.train_test_split(test_size=0.2)
    
    # Extract features and labels.
    x_train = data["train"]["features"]
    y_train = data["train"]["label"]
    x_test = data["test"]["features"]
    y_test = data["test"]["label"]
    

    return x_train, y_train, x_test, y_test

#used by XGboost
def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["features"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def load_data_XGBOOST(partition_id, num_partitions):
    """Load, preprocess, and partition the dataset."""
    data_directory = "C:\\Users\\brahim\\Desktop\\PFE2025\\Dataset\\csv\\"
    file_pattern = os.path.join(data_directory, "*.csv")
    data_files = glob(file_pattern)
    
    # Preprocess CSV files: read, concatenate, apply mapping, and drop the Activity column.
    df = preprocess_csv_files(data_files)
    
    # Drop rows where 'Stage' is missing so that features and labels have the same cardinality.
    df = df.dropna(subset=['Stage']).reset_index(drop=True)
    
    # Build the dataset:
    # All columns except 'Stage' are features.
    features = df.drop(columns=['Stage'])
    # The 'Stage' column is the label.
    target = df['Stage']
    
    # Convert to NumPy arrays with explicit dtypes.
    features_array = features.values.astype(np.float32)  # Convert features to float32.
    labels_array = target.values.astype(np.int8)          # Convert labels to int64.
    
    dataset_dict = {
        'features': features_array,
        'label': labels_array
    }
    
    print("Features shape:", features_array.shape)  
    # Create a Hugging Face Dataset from the dictionary.
    dataset = Dataset.from_dict(dataset_dict)
    
    # Create an IID partitioner and then a FederatedDataset.
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
        
    # Split the partition into 80% training and 20% testing.
    data = partition.train_test_split(test_size=0.2)
    
    # After creating the partitioned dataset:
    train_features = np.array(data["train"]["features"], dtype=np.float32)
    train_labels = np.array(data["train"]["label"], dtype=np.int8)
    test_features = np.array(data["test"]["features"], dtype=np.float32)
    test_labels = np.array(data["test"]["label"], dtype=np.int8)

    # Ensure features are 2D arrays
    if train_features.ndim == 1:
        train_features = train_features.reshape(-1, 1)
    if test_features.ndim == 1:
        test_features = test_features.reshape(-1, 1)

    # Create DMatrix with explicit features
    train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
    valid_dmatrix = xgb.DMatrix(test_features, label=test_labels)


    return train_dmatrix, valid_dmatrix

