from configparser import ConfigParser
import logging
logger = logging.getLogger(__name__)

parser = ConfigParser()
configfile = 'config/config.ini'
found = parser.read(configfile)
if not found:
    logger.error(f'{configfile} file not found')
    raise FileNotFoundError(f'{configfile} file not found')

class Config:
    """
    Handles the configuration input from config.ini file.
    """
    data_path = parser['traindataset']['path']
    data_filename = parser['traindataset']['filename']
    data_sep = parser['traindataset']['sep']
    data_loader = parser['traindataset']['library']
    data_label = parser['preprocessing']['label']
    data_scaler = parser['preprocessing']['scaler']
    testsize = float(parser['preprocessing']['test_size'])
    randomstate = int(parser['preprocessing']['random_state'])
    train_model_library = parser['model_trainer']['library']
    hyperparameter_optimizer_library = parser['optimizer']['library']
    output_path = parser['output']['path']
    output_filename = parser['output']['model_trained_name']
    output_loader = parser['output']['library']
    model_evaluator_library = parser['model_evaluator']['library']

    def __str__(self):
        attributes = []
        for attr, value in self.__class__.__dict__.items():
            if not attr.startswith("__"):
                attributes.append(f"{attr}: {value}")
        return "\n".join(attributes)


