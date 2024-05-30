from application.ports.ports import ModelTrainerInterface
from catboost import CatBoostRegressor
import logging

logger = logging.getLogger(__name__)

class CatBoostModelTrainer(ModelTrainerInterface):
    """CatboostRegressor implementation of  ports.ModelTrainerInterface."""
    def __init__(self):
        """Instantiate the model trainer with catboost.CatboostRegressor model."""
        self.model = CatBoostRegressor()

    def get_model(self):
        """
        Get the model.
        :return: model
        """
        return self.model

    def set_params(self, *args, **kwargs):
        """
        Set the model parameters.
        :param args:
        :param kwargs:
        :return: model
        """
        self.model = CatBoostRegressor(*args, **kwargs)
        return self.model

    def train(self, X, y, params:dict=None):
        """ Train the model with parameters."""
        logger.debug(f"Training model {self.model} with parameters:{params}...")
        if params:
            self.model = CatBoostRegressor(**params)
        cat_feats = [feat for feat in X.columns if X[feat].dtype == 'category']
        self.model.set_params(cat_features=cat_feats)
        return self.model.fit(X,y)

    def predict(self, X):
        """
        Predict with the model.
        :param X: features to evaluate
        :return: labels predicted
        """
        logger.debug(f"Testing model {self.model}...")
        return self.model.predict(X)