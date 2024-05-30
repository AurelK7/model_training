from abc import ABC, abstractmethod

class DataLoaderInterface(ABC):
    """
    DataLoader interface.
    """
    @abstractmethod
    def load(self):
        """
        Every implementation of a DataLoader should have this fct defined.
        """
        raise NotImplementedError


class PreprocessorInterface(ABC):
    """
    Preprocessor interface.
    """
    @abstractmethod
    def clean_data(self, data):
        """"
        data cleaning.
        this fct is optional.
        """
        pass

    @abstractmethod
    def split_data(self, X, y):
        """
        split data into train and test set.
        optional fct.
        :param X: features
        :param y: labels
        :return: train and test set for features and labels
        """
        pass

    @abstractmethod
    def scale_data(self, X_train, X_test):
        """
        scale data.
        optional fct.
        :param X_train:
        :param X_test:
        :return: fitted scaler on X_train and scaled data
        """
        pass

    @abstractmethod
    def format(self, data):
        """"
        Every implementation of a Preprocessor should have this fct defined.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scaler(self):
        """
        get fitted scaler.
        """
        pass


class ModelTrainerInterface(ABC):
    """"
    ModelTrainer interface.
    """
    @abstractmethod
    def get_model(self):
        pass

    def set_params(self):
        pass

    @abstractmethod
    def train(self, X, y, params:dict=None):
        """
        Every implementation of a ModelTrainer should have this fct defined.
        :param X: features
        :param y: labels
        :param params: parameters to use for model training. optinal params for the model.
        :return: trained model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """
        Every implementation of a ModelTrainer should have this fct defined.
        :param X: features to predict
        :return: prediction
        """
        raise NotImplementedError


class HyperparameterOptimizerInterface(ABC):
    """
    HyperparameterOptimizer interface.
    """
    @abstractmethod
    def optimize_hyperparameters(self, X, y):
        """
        Every implementation of a HyperparameterOptimizer should have this fct defined.
        :param X:
        :param y:
        :return: best parameters
        """
        raise NotImplementedError


class ModelSaverInterface(ABC):
    """
    ModelSaver interface.
    """
    @abstractmethod
    def save(self, model, savename=None):
        """
        Every implementation of a ModelSaver should have this fct defined.
        :param model: model to save
        :param savename: saving name. optional
        """
        raise NotImplementedError


class ModelEvaluatorInterface(ABC):
    """
    ModelEvaluator interface.
    """
    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Every implementation of a ModelEvaluator should have this fct defined.
        :param y_true: real labels
        :param y_pred: predicted labels
        :return: metrics result
        """
        raise NotImplementedError

