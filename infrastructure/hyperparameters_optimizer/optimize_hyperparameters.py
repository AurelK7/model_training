from typing import AnyStr
from application.ports.ports import HyperparameterOptimizerInterface
from infrastructure.hyperparameters_optimizer.optimizer.grid_search import GridSearchOptimizer
from infrastructure.hyperparameters_optimizer.optimizer.optuna import OptunaOptimizer
import logging

logger = logging.getLogger(__name__)

class Optimizer(HyperparameterOptimizerInterface):
    """Base implementation of a port.HyperparameterOptimizerInterface."""

    def __init__(self, library_name: AnyStr, model_trainer) -> None:
        self.library_name = library_name
        available_optimizer_library = {
            'gridsearch': GridSearchOptimizer,
            'optuna': OptunaOptimizer
        }
        # Raise keyError if optimizer not available
        try:
            self.optimizer = available_optimizer_library[self.library_name](model_trainer)
        except KeyError:
            logger.error(f"Library {self.library_name} is not available")
            raise Exception(f"Library {self.library_name} is not available")
    def optimize_hyperparameters(self, X, y):
        logger.debug(f"Calling optimize_hyperparameters with library: {self.library_name}...")
        return self.optimizer.optimize_hyperparameters(X, y)

