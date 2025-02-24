# client_XGBOOST.py

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
import xgboost as xgb
import numpy as np
from typing import Dict, Tuple
from fedsec.task import load_data_XGBOOST

class FlowerClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round=1,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 3,
            'learning_rate': 0.1,
        }

    def fit(self, ins: FitIns) -> FitRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        global_model_bytes = bytearray(ins.parameters.tensors[0])
        bst.load_model(global_model_bytes)

        # Compute predictions and gradients/hessians
        preds = bst.predict(self.train_dmatrix, output_margin=True)
        labels = self.train_dmatrix.get_label()
        grad = preds - labels  # First-order gradients
        hess = preds * (1.0 - preds)  # Second-order gradients

        # Convert to bytes for transmission
        grad_bytes = grad.tobytes()
        hess_bytes = hess.tobytes()
        
        if ins.config["global_round"] == "1":
            num_features = self.train_dmatrix.num_col()
            metrics = {"num_features": num_features}
        else:
            metrics = {}

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[grad_bytes, hess_bytes]),
            num_examples=self.num_train,
            metrics=metrics,  # Send feature count here
        )


    # Keep `evaluate()` unchanged
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate current model."""
        try:
            # Load current global model
            self.model = xgb.Booster(self.params)
            self.model.load_model(bytearray(ins.parameters.tensors[0]))

            # Evaluate model
            preds = self.model.predict(self.valid_dmatrix)
            labels = self.valid_dmatrix.get_label()

            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, preds)

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=0.0,  # Not used
                num_examples=self.num_val,
                metrics={"AUC": float(auc)}
            )
        except Exception as e:
            return EvaluateRes(
                status=Status(code=Code.ERROR, message=str(e)),
                loss=0.0,
                num_examples=0,
                metrics={}
            )

def client_fn(context: Context) -> Client:
    """Create a Flower client with gradient sharing capabilities."""
    # Load and partition data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    train_dmatrix, valid_dmatrix = load_data_XGBOOST(partition_id, num_partitions)
    
    # Create client instance
    return FlowerClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=train_dmatrix.num_row(),
        num_val=valid_dmatrix.num_row()
    )

app = ClientApp(client_fn=client_fn)