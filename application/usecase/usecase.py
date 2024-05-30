from domain import domain as domain
from time import sleep
import logging

logger = logging.getLogger(__name__)

class UseCase:
    def __init__(
            self,
            data_loader,
            preprocessor,
            train_model,
            model_saver,
            model_evaluator

    ) -> None:
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.train_model = train_model
        self.model_saver = model_saver
        self.model_evaluator = model_evaluator

    def execute(self) -> domain.Model:
        """
        Execute the usecase train model, save model and scaler and evaluate model
        :return: trained model
        """
        logger.debug(f"Running usecase excute...")
        data = self.data_loader.load()
        X_train, X_test, y_train, y_test = self.preprocessor.format(data)
        self.model_saver.save(self.preprocessor.get_scaler(), 'data_scaler.pkl')
        bestparams = self.train_model.optimize_hyperparameters(X_train, y_train)
        model = self.train_model.train(X_train, y_train, bestparams)
        self.model_saver.save(model)
        y_pred = model.predict(X_train)
        score_train = self.model_evaluator.score(y_train, y_pred)
        y_pred = model.predict(X_test)
        score_test = self.model_evaluator.score(y_test, y_pred)
        sleep(1)
        print(f"Train score: {score_train}, Test score: {score_test}")
        return model


