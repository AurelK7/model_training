from config.config import Config as config
from application.adapters import adapters as adapter
from infrastructure.data_loader.data_loader import DataLoader
from infrastructure.model_saver.save_model import ModelSaver
from domain.domain import Model as domain
from time import sleep
import logging

logger = logging.getLogger(__name__)


class Run:

    def execute(self):
        logger.debug("Program starting...")
        data = self.get_dataset()
        print(f"data shape: {data.shape}")
        print(data.head())
        sleep(3)
        X_train, X_test, y_train, y_test = self.get_preprocessor_format(data)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        sleep(5)
        model = self.get_model()
        # Optimize hyperparameters
        best_params = model.optimize_hyperparameters(X_train, y_train)
        print("Best hyperparameters: ", best_params)
        sleep(3)

        # Train the model with best parameters and provide the prediction
        model.train(X_train, y_train, best_params) #fit_intercept=best_params['fit_intercept'])

        print(f"Score on train: {model.score(X_train, y_train)}")
        print(f"Score on test: {model.score(X_test, y_test)}")
        sleep(3)
        # Save the model
        self.get_save_model().save(model)




    def get_dataset(self):
        """Load the dataset for training."""
        logger.debug("Loading the dataset...")
        dataloader = DataLoader(
            library_name = config.data_loader,
            data_path = config.data_path,
            data_filename = config.data_filename
        )
        return dataloader.load(sep=config.data_sep)

    def get_model(self) -> domain:
        """Instantiate  a Model for optimization and training."""
        return domain(
            model_trainer = self.get_model_trainer(),
            hyperparameter_optimizer =self.get_hyperparameter_optimizer()
        )

    @staticmethod
    def get_preprocessor_format(data):
        """Pre-process the dataset."""
        logger.debug("Start pre-processing of the dataset...")
        return adapter.ScikitLearnDataPreprocessor().format(
            data=data,
            y_label=config.data_label,
            test_size=config.testsize,
            random_state=config.randomstate
        )

    @staticmethod
    def get_hyperparameter_optimizer():
        """Optimize the hyperparameters."""
        return adapter.ScikitLearnGridSearchOptimizer()

    @staticmethod
    def get_model_trainer():
        """Train the model."""
        return adapter.ScikitLearnModelTrainer()

    @staticmethod
    def get_save_model():
        """Save the trained model."""
        logger.debug(f"Saving the trained model at {config.output_path} under {config.output_filename}...")
        model_saver = ModelSaver(
            library_name=config.output_loader,
            model_path=config.output_path,
            filename=config.output_filename
        )
        return model_saver #.save(model)





