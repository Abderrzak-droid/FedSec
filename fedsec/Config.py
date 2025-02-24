import os
import toml

def rename_files_and_update_toml(model_type):
    # Define the base directory where the files are located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the file names and TOML values based on the model type
    if model_type.lower() == "dnn":
        client_file = "client_app.py"
        server_file = "server_app.py"
        client_app_value = "fedsec.client_app:app"
        server_app_value = "fedsec.server_app:app"
    elif model_type.lower() == "xgboost":
        client_file = "client_XGBOOST.py"
        server_file = "server_XGBOOST.py"
        client_app_value = "fedsec.client_XGBOOST:app"
        server_app_value = "fedsec.server_XGBOOST:app"
    else:
        print("Invalid model type. Please choose 'DNN' or 'XGBoost'.")
        return

    
    # Update the pyproject.toml file
    toml_path = os.path.join(base_dir, "pyproject.toml")
    if os.path.exists(toml_path):
        with open(toml_path, "r") as f:
            config = toml.load(f)
        
        # Update the serverapp and clientapp values
        config["tool"]["flwr"]["app"]["components"]["serverapp"] = server_app_value
        config["tool"]["flwr"]["app"]["components"]["clientapp"] = client_app_value
        
        # Write the updated configuration back to the file
        with open(toml_path, "w") as f:
            toml.dump(config, f)
        print("Updated pyproject.toml with new serverapp and clientapp values.")

                # Prompt the user to update configuration values
        num_rounds = input("Enter the number of server rounds (or press Enter to keep current value): ")
        if num_rounds:
            config["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = int(num_rounds)
        
        num_clients = input("Enter the number of clients (or press Enter to keep current value): ")
        if num_clients:
            config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"] = int(num_clients)
        
        local_epochs = input("Enter the number of local epochs (or press Enter to keep current value): ")
        if local_epochs:
            config["tool"]["flwr"]["app"]["config"]["local-epochs"] = int(local_epochs)
        
        batch_size = input("Enter the batch size (or press Enter to keep current value): ")
        if batch_size:
            config["tool"]["flwr"]["app"]["config"]["batch-size"] = int(batch_size)
        
        # Write the updated configuration back to the file
        with open(toml_path, "w") as f:
            toml.dump(config, f)
        print("Updated pyproject.toml with new values.")
    else:
        print(f"pyproject.toml file not found at {toml_path}.")

if __name__ == "__main__":
    # Get the model type from the user
    model_type = input("Enter the model type (DNN or XGBoost): ")
    rename_files_and_update_toml(model_type)