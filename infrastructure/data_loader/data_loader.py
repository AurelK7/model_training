from typing import AnyStr
from infrastructure.data_loader.library import csv
from application.ports.ports import DataLoaderInterface
import logging
logger = logging.getLogger(__name__)

class DataLoader(DataLoaderInterface):
    """Base implementation of a port.DataLoaderInterface."""
    def __init__(self,
                 library_name: AnyStr,
                 data_path: AnyStr,
                 data_filename: AnyStr, *args, **kwargs) -> None:
        self.library_name = library_name
        available_library = {
            'csv': csv.DataLoader
        }
        # raise Keyerror if the library_name not available
        try:
            self.library = available_library[self.library_name](data_path, data_filename,  *args, **kwargs)
        except KeyError:
            logger.error(f"Library {self.library_name} is not available")
            raise KeyError(f"Library {self.library_name} is not available")

    def load(self):
        logger.debug(f"Calling library: {self.library_name} to load  dataset...")
        return self.library.load()


