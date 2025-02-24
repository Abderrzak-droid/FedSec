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
from sklearn.ensemble import RandomForestClassifier
tf.compat.v1.enable_v2_behavior()
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class RandomForestWrapper:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',  # Handle class imbalance
            max_depth=10,
            random_state=42
        )
        self.is_fitted = False

    def get_weights(self):
        if not self.is_fitted:
            return [np.array([], dtype=np.uint8)]
        # Serialize the model using pickle
        import pickle
        serialized_model = pickle.dumps(self.model)
        return [np.frombuffer(serialized_model, dtype=np.uint8)]

    def set_weights(self, weights):
        if len(weights[0]) == 0:
            return
        # Deserialize the model
        import pickle
        serialized_model = weights[0].tobytes()
        self.model = pickle.loads(serialized_model)
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
    if model_type == "randomforest": 
        scale_pos_weight = 75/25  # approximately 2.85 :  ratio of negative to positive samples

        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,
            random_state=42
        )
        return model
    
    model = tf.keras.models.Sequential([
        # Input layer with normalization
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        
        # First dense block
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Second dense block
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Third dense block
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
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
    
    # List of columns to keep
    wanted_columns = [
        'Src Port', 'Dst Port', 'Flow Duration', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Flow Packets/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Length',
        'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Std', 'Down/Up Ratio',
        'Bwd Init Win Bytes','Stage'
    ]
    
    for file in data_files:
        df = pd.read_csv(file)
        
        # Map the 'Stage' column using the provided mapping if it exists
        if 'Stage' in df.columns:
            df['Stage'] = df['Stage'].map(stage_mapping)
        
        # Filter the DataFrame to only include the wanted columns.
        # This automatically drops 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', etc.
        df = df[[col for col in df.columns if col in wanted_columns]]
        
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
    
    # Create a Hugging Face Dataset from the dictionary.
    dataset = Dataset.from_dict(dataset_dict)
    
    # Create an IID partitioner and then a FederatedDataset.
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
        
    # Split the partition into 80% training and 20% testing.
    data = partition.train_test_split(test_size=0.2)

    # Reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(data["train"])
    valid_dmatrix = transform_dataset_to_dmatrix(data['test'])

    return train_dmatrix, valid_dmatrix

