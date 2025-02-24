# server_XGBOOST.py

from typing import Dict, List, Tuple, Optional, Union
from flwr.common import (
    Parameters,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Context,
    EvaluateRes,
    FitIns,
    EvaluateIns,
    Scalar,
)
from flwr.server.strategy import Strategy
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
import numpy as np
import xgboost as xgb

class FedXgbGradient(Strategy):
    def __init__(self, proxy_data, **kwargs):
        super().__init__(**kwargs)
        self.proxy_data = proxy_data
        self.global_model = None  # Holds the global XGBoost model

    def aggregate_fit(self, server_round, results, failures):
        # Aggregate gradients/hessians from clients
        total_samples = sum([r.num_examples for r in results])
        grad_list, hess_list = [], []
        if server_round == 1:
            self.num_features = int(next(iter(results)).metrics["num_features"])
            assert self.num_features > 0, "No features detected from clients!" 
            
            # Initialize model with dummy data (correct features)
            dummy_data = np.zeros((1, self.num_features))
            dummy_dmatrix = xgb.DMatrix(dummy_data)
            self.global_model = xgb.train({}, dummy_dmatrix, num_boost_round=0)

        for result in results:
            grad = np.frombuffer(result.parameters.tensors[0], dtype=np.float32)
            hess = np.frombuffer(result.parameters.tensors[1], dtype=np.float32)
            grad_list.append(grad * (result.num_examples / total_samples))  # Weighted average
            hess_list.append(hess * (result.num_examples / total_samples))

        global_grad = np.sum(grad_list, axis=0)
        global_hess = np.sum(hess_list, axis=0)

        # Update proxy dataset with aggregated gradients/hessians
        self.proxy_data.set_info(gradient=global_grad, hessian=global_hess)

        # Train a new tree using the proxy data
        if self.global_model is None:
            bst = xgb.train({}, self.proxy_data, num_boost_round=1)
        else:
            bst = self.global_model
            bst = xgb.train({}, self.proxy_data, num_boost_round=1, xgb_model=bst)

        self.global_model = bst

        # Serialize the updated model
        model_bytes = bst.save_raw("json")
        return Parameters(tensors=[model_bytes])

    def configure_fit(self, server_round, parameters, client_manager):
        # Send the current global model to clients
        return super().configure_fit(server_round, parameters, client_manager)
    """Strategy for Federated XGBoost with gradient sharing."""
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        
        # Initialize XGBoost parameters
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 3,
            'learning_rate': 0.1,
            'tree_method': 'hist'
        }
        self.model = None

    def initialize_parameters(
        self, client_manager: ClientProxy
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        self.model = xgb.Booster(self.params)
        model_bytes = self.model.save_raw()
        return Parameters(tensors=[model_bytes])

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientProxy
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {"global_round": str(server_round)}
        fit_ins = FitIns(parameters, config)

        # Sample clients for fitting
        sample_size = int(self.fraction_fit * len(client_manager.all()))
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientProxy
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Sample clients for evaluation
        sample_size = int(self.fraction_evaluate * len(client_manager.all()))
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )
        
        config = {"global_round": str(server_round)}
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_gradients(self, gradients: List[NDArrays]) -> NDArrays:
        """Aggregate gradients from multiple clients."""
        # Extract gradients and hessians
        grads = []
        hess = []
        for grad_array in gradients:
            grads.append(grad_array[0][0])  # First order gradients
            hess.append(grad_array[0][1])   # Second order gradients

        # Average the gradients
        avg_grad = np.mean(grads, axis=0)
        avg_hess = np.mean(hess, axis=0)

        return [np.array([avg_grad, avg_hess])]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results from clients."""
        if not results:
            return None, {}

        # Extract gradients from client results
        gradients = [fit_res.parameters.tensors for _, fit_res in results]
        aggregated_gradients = self.aggregate_gradients(gradients)

        if server_round == 1:
            # Initialize model on first round
            self.model = xgb.Booster(self.params)
        
        # Update model with aggregated gradients
        grad = aggregated_gradients[0][0]
        hess = aggregated_gradients[0][1]
        
        # Create DMatrix with aggregated gradients
        num_samples = len(grad)
        dtrain = xgb.DMatrix(
            np.zeros((num_samples, 1)),  # Dummy features
            label=np.zeros(num_samples)   # Dummy labels
        )
        dtrain.set_base_margin(np.zeros(num_samples))
        
        # Set gradients
        dtrain.set_info(grad=grad, hess=hess)
        
        # Update model
        self.model.update(dtrain, server_round)
        
        # Get updated model parameters
        model_bytes = self.model.save_raw()
        
        return Parameters(tensors=[model_bytes]), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}
        
        # Call aggregate metrics function
        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )
            return 0.0, metrics_aggregated

        return 0.0, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Optional server-side evaluation function."""
        return None

def server_fn(context: Context) -> ServerAppComponents:
    """Create server instance with gradient sharing strategy."""
    # Get configuration
    num_rounds = context.run_config.get("num-server-rounds", 3)
    
    # Create strategy
    strategy = FedXgbGradient(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average_metrics
    )
    
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )

def weighted_average_metrics(metrics):
    """Aggregate evaluation metrics weighted by number of samples."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    weighted_auc = sum(
        metrics_dict["AUC"] * num_samples
        for num_samples, metrics_dict in metrics
    ) / total_samples
    return {"AUC": weighted_auc}

# Create ServerApp
app = ServerApp(server_fn=server_fn)