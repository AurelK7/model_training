from pandas import read_csv
from application.ports.ports import DataLoaderInterface
import logging
logger = logging.getLogger(__name__)


class DataLoader(DataLoaderInterface):
    """
    Load a dataset with pandas read_csv.
    """
    def __init__(self, data_path , data_filename,  *args, **kwargs) -> None:
        """"
        Instantiate the pandas read_csv with the path to the dataset and the filename.
        """
        self.path = data_path
        self.filename = data_filename
        self.args = args
        self.kwargs = kwargs

    def load(self):
        """Load the dataset with pandas read_csv."""
        logger.debug(f"Pandas read_csv loading {self.filename} dataset...")
        path = self.path + '/'+ self.filename
        return read_csv(path, *self.args, **self.kwargs )

