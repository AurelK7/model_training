from typing import AnyStr
from infrastructure.model_saver.library import pickle as pkl
from application.ports.ports import ModelSaverInterface
import logging
logger = logging.getLogger(__name__)

class ModelSaver(ModelSaverInterface):
    """
    Base implementation of ports.ModelSaverInterface
    """
    def __init__(
            self,
            library_name: AnyStr,
            model_path: AnyStr ,
            filename:AnyStr
    ) -> None:
        self.library_name = library_name
        available_library = {
            'pickle': pkl.ModelSaver
        }
        # raise keyError if library name if not available
        try:
            self.library = available_library[self.library_name](model_path, filename)
        except:
            logger.error(f"Library {self.library_name} not available")
            raise Exception(f"Library {self.library_name} not available")

    def save(self, model, savename=None):
        logger.debug(f"Saving model with library {self.library_name}...")
        return self.library.save(model, savename)

