from config.config import Config as config
from infrastructure.data_loader.data_loader import DataLoader
from infrastructure.preprocessor.preprocessor import DataPreprocessor
from infrastructure.model_trainer.train_model import Trainer
from infrastructure.hyperparameters_optimizer.optimize_hyperparameters import Optimizer
from infrastructure.model_saver.save_model import ModelSaver
from infrastructure.model_evaluator.evaluate_model import ModelEvaluator
from domain import domain
from application.usecase.usecase import UseCase
import logging

logger = logging.getLogger(__name__)


class Container:
    """
    Dependency injection container.

    This object contains every instances required by the config.

    You will define here what implementations should be used to :

    - train a  model as describe in domain (optimizer, trainer, saver, evaluator)
    - ...
    """
    train_usecase: UseCase
    def __init__(self, dataset:domain.Dataset =None):
        logger.debug(f"Running with configurations input parameters:\n{config()}")
        self.dataset = dataset
        self.train_usecase = self.get_usecase()


    def get_usecase(self) -> UseCase:
        """Instantiate UseCase with config parameters."""
        logger.debug("Running get_usecase...")
        return UseCase(
            data_loader=self.get_dataset(),
            preprocessor=self.get_preprocessor(),
            train_model=self.get_model(),
            model_saver=self.get_save_model(),
            model_evaluator=self.get_model_evaluator()
        )

    def get_dataset(self) -> DataLoader:
        """Instantiate DataLoader with config parameters."""
        logger.debug("Running get_dataset...")
        if self.dataset is None:
            dataloader = DataLoader(
                library_name = config.data_loader,
                data_path = config.data_path,
                data_filename = config.data_filename,
                sep = config.data_sep
            )
            return dataloader
        else:
            dataloader = DataLoader(**self.dataset.items)
            return dataloader

    @staticmethod
    def get_preprocessor() -> DataPreprocessor:
        """Instantiate DataPreprocessor with config parameters. """
        logger.debug("Running get_preprocessing...")
        return DataPreprocessor(
            library_name = config.data_scaler,
            y_label = config.data_label,
            test_size = config.testsize,
            random_state = config.randomstate
        )

    def get_model(self) -> domain.Model:
        """Instantiate  a domain.Model for optimization and training."""
        logger.debug("Running get_model...")
        return domain.Model(
            model_trainer = self.get_model_trainer(),
            #hyperparameter_optimizer =self.get_hyperparameter_optimizer()
        )

    @staticmethod
    def get_model_trainer() -> Trainer:
        """Instantiate Train the model."""
        logger.debug("Running get_model_trainer...")
        return Trainer(
            library_name=config.train_model_library
        )

    def get_hyperparameter_optimizer(self) -> Optimizer:
        """Instantiate HyperparameterOptimizer with config parameters."""
        logger.debug("Running get_hyperparameter_optimizer...")
        return Optimizer(
            library_name=config.hyperparameter_optimizer_library,
            model_trainer=self.get_model_trainer().get_model()
        )

    @staticmethod
    def get_save_model() -> ModelSaver:
        """Instantiate ModelSaver with config parameters."""
        logger.debug(f"Saving the trained model at {config.output_path} under {config.output_filename}...")
        return ModelSaver(
            library_name=config.output_loader,
            model_path=config.output_path,
            filename=config.output_filename
        )
    @staticmethod
    def get_model_evaluator() -> ModelEvaluator:
        """Instantiate ModelEvaluator with config parameters."""
        logger.debug("Running get_model_evaluator...")
        library_name=config.model_evaluator_library
        library_name =[lib.lstrip() for lib in library_name.split(',')]
        return ModelEvaluator(
            library_name=library_name
        )