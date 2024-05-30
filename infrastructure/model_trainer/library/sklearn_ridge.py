from application.ports.ports import ModelTrainerInterface
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)

class LinearRidgeModelTrainer(ModelTrainerInterface):
    """Linear Ridge implementation of ports.ModelTrainerInterface."""
    def __init__(self):
        """Instantiate the model trainer with sklearn linear model Ridge."""
        self.model = Ridge()

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
        self.model = Ridge(*args, **kwargs)
        return self.model

    def train(self, X, y, params:dict=None):
        """ Train the model with parameters."""
        logger.debug(f"Training model {self.model} with parameters:{params}...")
        if params:
            self.model = Ridge(**params)
        return self.model.fit(X,y)

    def predict(self, X):
        """
        Predict with the model.
        :param X: features to evaluate
        :return: labels predicted
        """
        logger.debug(f"Testing model {self.model}...")
        return self.model.predict(X)