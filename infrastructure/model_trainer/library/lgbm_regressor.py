from application.ports.ports import ModelTrainerInterface
from lightgbm import LGBMRegressor
import logging

logger = logging.getLogger(__name__)

class LgbmModelTrainer(ModelTrainerInterface):
    """LightGbmRegressor implementation of ports.ModelTrainerInterface."""
    def __init__(self):
        """Instantiate the model trainer with lightgbm.LgbmRegressor model."""
        self.model = LGBMRegressor()

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
        self.model = LGBMRegressor(*args, **kwargs)
        return self.model

    def train(self, X, y, params:dict=None):
        """ Train the model with parameters."""
        logger.debug(f"Training model {self.model} with parameters:{params}...")
        if params:
            self.model = LGBMRegressor(**params)
        cat_features = [feat for feat in X.columns if X[feat].dtype=='category']
        return self.model.fit(X,y, categorical_feature=cat_features)

    def predict(self, X):
        """
        Predict with the model.
        :param X: features to evaluate
        :return: labels predicted
        """
        logger.debug(f"Testing model {self.model}...")
        return self.model.predict(X)