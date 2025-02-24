from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import numpy as np
from logging import getLogger
from fedsec.task import load_model, load_data
from datasets.utils.logging import enable_progress_bar
enable_progress_bar()
logger = getLogger(__name__)

class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        try:
            # Convert data to correct format if needed
            self.x_train = np.array(self.x_train, dtype=np.float32)
            self.y_train = np.array(self.y_train, dtype=np.int8)
            
            # Set model weights
            self.model.set_weights(parameters)
            
            # Calculate class weights
            neg_class = np.sum(self.y_train == 0)
            pos_class = np.sum(self.y_train == 1)
            total = neg_class + pos_class
            
            # Create class weight dictionary
            class_weight = {
                0: (1 / neg_class) * (total / 2),
                1: (1 / pos_class) * (total / 2)
            }
            # Train the model
            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_split=0.1,  # Add validation to monitor training
                class_weight=class_weight,
            )
            
            
            
            logger.info(f"Training finished with history: {history.history}")
            
            return self.model.get_weights(), len(self.x_train), {
                "loss": history.history['loss'][-1],
                "accuracy": history.history['accuracy'][-1]
            }
        except Exception as e:
            logger.error(f"Error in fit method: {str(e)}")
            raise

    def evaluate(self, parameters, config):
        try:
            # Convert data to correct format if needed
            self.x_test = np.array(self.x_test, dtype=np.float32)
            self.y_test = np.array(self.y_test, dtype=np.int32)
            
            # Set model weights
            self.model.set_weights(parameters)
            
            # Evaluate the model
            loss, accuracy = self.model.evaluate(
                self.x_test, 
                self.y_test, 
                verbose=0
            )
            
            logger.info(f"Evaluation completed - Loss: {loss}, Accuracy: {accuracy}")
            
            return loss, len(self.x_test), {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"Error in evaluate method: {str(e)}")
            raise

def client_fn(context: Context):
    try:
        # Load model and data
        input_shape = (20,)  # Make sure this matches your data dimensions

        # In server_app.py or client_app.py
        net = load_model(input_shape,"DNN")
        
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions)
        
        # Get training parameters from config
        epochs = context.run_config.get("local-epochs", 1)  # Default to 1 if not specified
        batch_size = context.run_config.get("batch-size", 32)  # Default to 32 if not specified
        verbose = context.run_config.get("verbose", 0)  # Default to 0 if not specified
        
        logger.info(f"Creating client with partition {partition_id}/{num_partitions}")
        
        # Return Client instance
        return FlowerClient(
            net, data, epochs, batch_size, verbose
        ).to_client()
    except Exception as e:
        logger.error(f"Error in client_fn: {str(e)}")
        raise

# Flower ClientApp
app = ClientApp(client_fn=client_fn)