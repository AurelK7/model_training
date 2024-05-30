from application.ports.ports import HyperparameterOptimizerInterface
from sklearn.model_selection import GridSearchCV
from numpy.random import uniform, randint
import logging

logger = logging.getLogger(__name__)

class GridSearchOptimizer(HyperparameterOptimizerInterface):
    """
    GridSearchCV implementation of ports.HyperparameterOptimizerInterface.
    """
    def __init__(self, model_trainer=None) -> None:
        """
        Instantiate GridSearchCV with the model to optimize
        :param model_trainer:
        """
        self.model = model_trainer
    def optimize_hyperparameters(self, X, y):
        """
        Optimize hyperparameters of the model with GridSearchCV.
        Need to define what parameters to optimize in the param_grid.
        :param X:
        :param y:
        :return: best parameters
        """
        logger.debug(f"Optimizing hyperparameters of model {self.model} with GridSearchCV...")
        print("Optimizer is running, please wait...")
        param_grid = {
            'fit_intercept': [True, False],
            'max_iter': randint(50,10000, 10),
            'alpha': uniform(1e-10, 10,10),
            'tol': uniform(1e-10, 1e-3, 10)
        }
        grid_search = GridSearchCV(self.model,
                                   param_grid,
                                   cv=5,
                                   n_jobs=-1,
                                   scoring='r2',
                                   verbose=1)
        grid_search.fit(X, y)
        logger.debug(f"best_estimator: {grid_search.best_estimator_} with score {grid_search.best_score_} "
                   )
        return grid_search.best_params_

