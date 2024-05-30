import pickle
from application.ports.ports import ModelSaverInterface
import logging
logger = logging.getLogger(__name__)


class ModelSaver(ModelSaverInterface):
    """saving the trained model using pickle."""

    def __init__(self, savepath, filename ):
        """Instantiate the model saver."""
        self.savepath = savepath
        self.filename = filename

    def save(self, model, savename=None):
        """
        Save the trained model.
        :param model: model to save
        :param savename: saving name. optional
        """
        # create the directory if it does not exist
        name = savename if savename else self.filename
        fullpath = self.savepath + '/' + name if self.savepath else name
        from pathlib import Path
        Path(self.savepath).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Pickle saving model at {fullpath}...")
        return pickle.dump(model, open(fullpath, 'wb'))

