from application.ports.ports import HyperparameterOptimizerInterface
import optuna
from sklearn.model_selection import cross_val_score
from numpy.random import uniform, randint
import logging

logger = logging.getLogger(__name__)

class OptunaOptimizer(HyperparameterOptimizerInterface):
    """
    GridSearchCV implementation of HyperparameterOptimizerInterface
    """
    def __init__(self, model_trainer=None) -> None:
        """
        Instantiate GridSearchCV with the model to optimize
        :param model_trainer:
        """
        self.model = model_trainer
    def optimize_hyperparameters(self, X, y):
        """
        Optimize hyperparameters of the model with optuna.
        Need to define what parameters to optimize in the params.
        :param X:
        :param y:
        :return: best parameters
        """
        logger.debug(f"Optimizing hyperparameters of model {self.model} with Optuna...")
        print("Optimizer is running, please wait...")

        def objective(trial):
            params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'max_iter': trial.suggest_int('max_iter', 50, 10000),
                'alpha': trial.suggest_uniform('alpha', 1e-10, 10),
                'tol': trial.suggest_uniform('tol', 1e-10, 1e-3)
            }
            model = self.model(**params)
            scores = cross_val_score(
                                    estimator=model,
                                    X=X,
                                    y=y,
                                    cv=5,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=-1,

            )
            score = scores.mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        return study.best_params_

