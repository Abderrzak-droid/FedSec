from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdagrad, FedAvg, FedMedian, FedProx
from logging import getLogger
from fedsec.task import load_model
from datasets.utils.logging import enable_progress_bar
import os
import tensorflow as tf
from datetime import datetime
enable_progress_bar()

logger = getLogger(__name__)

class SaveModelStrategy(FedAvg):
    def __init__(
        self,
        *args,
        model=None,
        save_dir="saved_models",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.save_dir = save_dir
        self.best_accuracy = 0.0
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def aggregate_fit(self, server_round, results, failures):
        # Aggregate weights using parent class method
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Handle the parameters correctly
            if isinstance(aggregated_parameters, Parameters):
                weights = parameters_to_ndarrays(aggregated_parameters)
            else:
                weights = aggregated_parameters
            
            # Update model with aggregated parameters
            self.model.set_weights(weights)
            
            # Save model after each round with .keras extension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                self.save_dir, 
                f"model_round_{server_round}_{timestamp}.keras"  # Added .keras extension
            )
            self.model.save(model_path)
            logger.info(f"Saved model for round {server_round} at {model_path}")
        
        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Aggregate metrics using parent class method
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated is not None:
            loss, metrics = aggregated
            accuracy = metrics.get("accuracy", 0.0)
            
            # Save best model based on accuracy with .keras extension
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                best_model_path = os.path.join(self.save_dir, "best_model.keras")  # Added .keras extension
                self.model.save(best_model_path)
                logger.info(
                    f"New best model saved with accuracy {accuracy} at {best_model_path}"
                )
        
        return aggregated

def server_fn(context: Context):
    try:
        # Read from config
        num_rounds = context.run_config.get("num-server-rounds", 3)
        input_shape = (20,)
        save_dir = "E:\\PFE2025\\fedsec\\Attacker_DNN"
        
        # Initialize model
        initial_model = load_model(input_shape=input_shape, model_type="DNN")
        parameters = ndarrays_to_parameters(initial_model.get_weights())
        
        # Define strategy with model saving capability
        strategy = SaveModelStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            min_evaluate_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            model=initial_model,
            save_dir=save_dir
        )
        
        config = ServerConfig(num_rounds=num_rounds)
        logger.info(f"Server starting with {num_rounds} rounds")
        
        return ServerAppComponents(strategy=strategy, config=config)
    except Exception as e:
        logger.error(f"Error in server_fn: {str(e)}")
        raise

def weighted_average(metrics):
    """Aggregate evaluation metrics weighted by number of samples."""
    try:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        return {
            "accuracy": sum(accuracies) / sum(examples)
        }
    except Exception as e:
        logger.error(f"Error in metric aggregation: {str(e)}")
        raise

# Create ServerApp
app = ServerApp(server_fn=server_fn)