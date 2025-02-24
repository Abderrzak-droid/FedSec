from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from logging import getLogger
from fedsec.task_RF import load_model
from datasets.utils.logging import enable_progress_bar
import pickle

logger = getLogger(__name__)

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from logging import getLogger

logger = getLogger(__name__)

class FedAvgRF(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate Random Forest models from multiple clients."""
        if not results:
            return None, {}

        # Extract models from results
        models = []
        for _, fit_res in results:
            try:
                model_bytes = fit_res.parameters.tensors[0]
                model = pickle.loads(model_bytes)
                models.append(model)
            except Exception as e:
                logger.error(f"Error unpickling model: {str(e)}")
                continue

        if not models:
            return None, {}

        # Aggregate Random Forests
        aggregated_model = self.aggregate_random_forests(models)

        # Serialize the aggregated model
        try:
            aggregated_bytes = pickle.dumps(aggregated_model)
            parameters = Parameters(tensor_type="", tensors=[aggregated_bytes])
            return parameters, {}
        except Exception as e:
            logger.error(f"Error serializing aggregated model: {str(e)}")
            return None, {}

    def aggregate_random_forests(self, models: List[RandomForestClassifier]) -> RandomForestClassifier:
        """Aggregate multiple Random Forest models by combining their trees."""
        if not models:
            return None

        # Create a new Random Forest with all trees from all models
        base_model = models[0]
        all_estimators = []
        
        # Collect all trees from all models
        for model in models:
            all_estimators.extend(model.estimators_)

        # Create a new Random Forest with the combined trees
        aggregated_model = RandomForestClassifier(
            n_estimators=len(all_estimators),
            max_depth=base_model.max_depth,
            class_weight=base_model.class_weight,
            random_state=base_model.random_state
        )
        
        # Copy all necessary attributes from the base model
        attributes_to_copy = [
            'classes_',
            'n_classes_',
            'n_features_in_',
            'n_outputs_',
            'feature_names_in_',
            'max_features_',
            'n_features_',
            'criterion',
            'max_depth',
            'min_samples_split',
            'min_samples_leaf',
            'min_weight_fraction_leaf',
            'max_features',
            'max_leaf_nodes',
            'min_impurity_decrease',
            'class_weight',
            'random_state'
        ]
        
        for attr in attributes_to_copy:
            if hasattr(base_model, attr):
                setattr(aggregated_model, attr, getattr(base_model, attr))
        
        # Set the combined trees as the estimators
        aggregated_model.estimators_ = all_estimators
        
        # Initialize n_outputs_ if not set
        if not hasattr(aggregated_model, 'n_outputs_'):
            aggregated_model.n_outputs_ = 1  # For binary classification
            
        return aggregated_model
    

def get_initial_parameters(model):
    """Get initial parameters for the global model."""
    try:
        model_bytes = pickle.dumps(model)
        return Parameters(tensor_type="", tensors=[model_bytes])
    except Exception as e:
        logger.error(f"Error getting initial parameters: {str(e)}")
        raise

def weighted_average(metrics):
    """Aggregate evaluation metrics weighted by number of samples."""
    try:
        if not metrics:
            return {}
        
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        if not examples:
            return {}
            
        return {
            "accuracy": sum(accuracies) / sum(examples)
        }
    except Exception as e:
        logger.error(f"Error in metric aggregation: {str(e)}")
        return {}

def server_fn(context: Context):
    try:
        # Read from config
        num_rounds = context.run_config.get("num-server-rounds", 3)
        input_shape = (20,)
        
        # Initialize model
        initial_model = load_model(input_shape=input_shape, model_type="randomforest")
        
        # Get initial parameters
        initial_parameters = get_initial_parameters(initial_model)
        
        # Define strategy with model saving capability
        strategy = FedAvgRF(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            min_evaluate_clients=2,
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        
        config = ServerConfig(num_rounds=num_rounds)
        logger.info(f"Server starting with {num_rounds} rounds")
        
        return ServerAppComponents(strategy=strategy, config=config)
    except Exception as e:
        logger.error(f"Error in server_fn: {str(e)}")
        raise

# Create ServerApp
app = ServerApp(server_fn=server_fn)