from typing import AnyStr
from application.ports.ports import ModelTrainerInterface
from infrastructure.model_trainer.library.sklearn_ridge import LinearRidgeModelTrainer
from infrastructure.model_trainer.library.lgbm_regressor import LgbmModelTrainer
from infrastructure.model_trainer.library.catboost_regressor import CatBoostModelTrainer
import logging

logger = logging.getLogger(__name__)

class Trainer(ModelTrainerInterface):
    """Base implementation of a port.ModelTrainerInterface."""

    def __init__(self,library_name: AnyStr) -> None:
        self.library_name = library_name
        available_library = {
            'linear_ridge': LinearRidgeModelTrainer,
            'lgbm_regressor': LgbmModelTrainer,
            'catboost_regressor': CatBoostModelTrainer
        }
        try:
            self.model = available_library[self.library_name]()
        except KeyError:
            logger.error(f"Library {self.library_name} is not available")
            raise Exception(f"Library {self.library_name} is not available")

    def get_model(self):
        """ Mehtod to retrieve the trained model"""
        return self.model.get_model()

    def set_params(self, *args, **kwargs):
        """ Method for settings model parameters with external values"""
        return self.model.set_params(*args, **kwargs)

    def train(self, X, y, params:dict=None):
        """ Train method """
        logger.debug(f"Calling train with library:{self.library_name}...")
        return self.model.train(X,y, params)

    def predict(self, X):
        """ predict method"""
        logger.debug(f"Calling predict with library:{self.library_name}...")
        return self.model.predict(X)
