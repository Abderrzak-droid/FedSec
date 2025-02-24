from fedsec.task_RF import load_data
from flwr.client import Client, ClientApp
from flwr.common import (
    Parameters,
    Config,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Status,
    Code,
    Context
)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import logging as logger

class FlowerClient(Client):
    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        num_train,
        num_val
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.num_train = num_train
        self.num_val = num_val
        self.params = {
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',
            'random_state': 42
        }
        self.model = RandomForestClassifier(**self.params)
        # Fit the model immediately to initialize all attributes
        self.model.fit(self.x_train, self.y_train)

    def fit(self, ins: FitIns) -> FitRes:
        try:
            # If not first round, load global model
            if len(ins.parameters.tensors[0]) > 0:
                received_model = pickle.loads(ins.parameters.tensors[0])
                # Copy necessary attributes
                self.model = received_model

            # Train the model
            self.model.fit(self.x_train, self.y_train)

            # Serialize the model
            model_bytes = pickle.dumps(self.model)

            # Calculate metrics
            train_accuracy = self.model.score(self.x_train, self.y_train)
            
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=Parameters(tensor_type="", tensors=[model_bytes]),
                num_examples=self.num_train,
                metrics={"accuracy": float(train_accuracy)}
            )
        except Exception as e:
            logger.error(f"Error during fit: {str(e)}")
            return FitRes(
                status=Status(code=Code.GET_PARAMETERS_NOT_IMPLEMENTED, message=str(e)),
                parameters=Parameters(tensor_type="", tensors=[]),
                num_examples=0,
                metrics={}
            )
        
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        try:
            # Load global model
            self.model = pickle.loads(ins.parameters.tensors[0])

            # Evaluate the model
            accuracy = self.model.score(self.x_val, self.y_val)

            # Calculate predictions for AUC
            y_pred_proba = self.model.predict_proba(self.x_val)[:, 1]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(self.y_val, y_pred_proba)

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=1.0 - accuracy,  # Use 1-accuracy as a proxy for loss
                num_examples=self.num_val,
                metrics={
                    "accuracy": float(accuracy),
                    "auc": float(auc)
                }
            )
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return EvaluateRes(
                status=Status(code=Code.GET_PARAMETERS_NOT_IMPLEMENTED, message=str(e)),
                loss=float("inf"),
                num_examples=0,
                metrics={}
            )

def client_fn(context: Context) -> Client:
    try:
        # Load data
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        
        # Load data using the regular load_data function
        x_train, y_train, x_val, y_val = load_data(partition_id, num_partitions)
        
        # Get number of samples
        num_train = len(x_train)
        num_val = len(x_val)
        
        # Create and return client
        return FlowerClient(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_train=num_train,
            num_val=num_val
        )
    except Exception as e:
        logger.error(f"Error in client_fn: {str(e)}")
        raise

# Flower ClientApp
app = ClientApp(client_fn=client_fn)