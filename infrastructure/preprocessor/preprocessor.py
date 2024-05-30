from typing import AnyStr
from application.ports.ports import PreprocessorInterface
from infrastructure.preprocessor.scaler import standard
from infrastructure.preprocessor.scaler import vr
import logging
logger = logging.getLogger(__name__)

class DataPreprocessor(PreprocessorInterface):
    """
    Base implementation of ports.PreprocessorInterface.
    """
    def __init__(self,
                library_name: AnyStr,
                y_label:AnyStr,
                test_size: float=0.2,
                random_state: int = 42
                )-> None:
        self.library_name = library_name
        available_library = {
            'StandardScaler': standard.StandardDataPreprocessor,
            'vr': vr.DataPreprocessor
        }
        try:
            self.preprocessor = available_library[self.library_name](y_label, test_size, random_state)
        except KeyError:
            logger.error(f"Library {self.library_name} is not available")
            raise KeyError(f"Library {self.library_name} is not available")

    def clean_data(self, data):
        logger.debug(f"Calling clean_data with library: {self.library_name}...")
        return self.preprocessor.clean_data(data)

    def split_data(self, X, y):
        logger.debug(f"Calling split_data with library: {self.library_name}...")
        return self.preprocessor.split_data(X, y)

    def scale_data(self, X_train, X_test):
        logger.debug(f"Calling scale_data with library: {self.library_name}...")
        return self.preprocessor.scale_data(X_train, X_test)

    def format(self, data):
        logger.debug(f"Calling format with library: {self.library_name}...")
        return self.preprocessor.format(data)

    def get_scaler(self):
        logger.debug(f"Calling get_scaler with library: {self.library_name}...")
        return self.preprocessor.get_scaler()