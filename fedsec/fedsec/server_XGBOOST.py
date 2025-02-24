

from datetime import datetime
import os
from typing import Dict
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdagrad, FedAvg, FedMedian, FedProx, FedXgbBagging
from logging import getLogger

import io
import xgboost as xgb

logger = getLogger(__name__)

import json
from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy



class FedXgbBaggingSave(FedAvg):
    """Configurable FedXgbBagging strategy implementation."""

    def __init__(
        self,
        save_dir: str = "E:\\PFE2025\\fedsec\\Attacker_XGBoost",
        evaluate_function: Optional[
            Callable[
                [int, Parameters, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        **kwargs: Any,
    ):
        self.evaluate_function = evaluate_function
        self.global_model: Optional[bytes] = None
        self.save_dir = save_dir
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using bagging."""
        def aggregate(
            bst_prev_org: Optional[bytes],
            bst_curr_org: bytes,
        ) -> bytes:
            """Conduct bagging aggregation for given trees."""
            if not bst_prev_org:
                return bst_curr_org

            # Get the tree numbers
            tree_num_prev, _ = _get_tree_nums(bst_prev_org)
            _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

            bst_prev = json.loads(bytearray(bst_prev_org))
            bst_curr = json.loads(bytearray(bst_curr_org))

            bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ] = str(tree_num_prev + paral_tree_num_curr)
            iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
                "iteration_indptr"
            ]
            bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
                iteration_indptr[-1] + paral_tree_num_curr
            )

            # Aggregate new trees
            trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
            for tree_count in range(paral_tree_num_curr):
                trees_curr[tree_count]["id"] = tree_num_prev + tree_count
                bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
                    trees_curr[tree_count]
                )
                bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

            bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

            return bst_prev_bytes
        

        def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
            xgb_model = json.loads(bytearray(xgb_model_org))
            # Get the number of trees
            tree_num = int(
                xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                    "num_trees"
                ]
            )
            # Get the number of parallel trees
            paral_tree_num = int(
                xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                    "num_parallel_tree"
                ]
            )
            return tree_num, paral_tree_num
        

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = self.global_model
        for _, fit_res in results:
            update = fit_res.parameters.tensors
            for bst in update:
                global_model = aggregate(global_model, bst)

        self.global_model = global_model
                    # Save model after each round with .keras extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.save_dir, 
            f"model_round_{server_round}_{timestamp}.model"  # Added .keras extension
        )
        with open(model_path, "wb") as f:
            f.write(self.global_model)


        return (
            Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
            {},
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics using average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return 0, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_function is None:
            # No evaluation function provided
            return None
        eval_res = self.evaluate_function(server_round, parameters, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

def server_fn(context: Context) -> ServerAppComponents:
    try:
        # Read from config
        num_rounds = context.run_config.get("num-server-rounds", 3)
        
        # Define strategy
        strategy = FedXgbBaggingSave(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=1,  # Minimum number of clients for training
            min_evaluate_clients=1,  # Minimum number of clients for evaluation
            min_available_clients=1,  # Minimum number of total clients in the system
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
        )
        
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds)
        )
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

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config

# Create ServerApp
app = ServerApp(server_fn=server_fn)

