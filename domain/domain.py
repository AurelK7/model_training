
class Dataset:
    """
    Define a model of dataset as expected by data loader.
    It is used to pass the dataset information to the usecase via the container. If you provide Dataset to
    container, only this dataset information will use for data_loader, instead of information from config file
    example:
        container(Dataset('data', 'data.csv', 'library')).get_usecase().execute()
    """

    def __init__(self, path, filename, library) -> None:
        self.items = {
            'data_path': path,
            'data_filename': filename,
            'library_name': library
        }
        self.path = path
        self.filename = filename

    def __getitem__(self, index):
        return self.items[index]


class Model:
    """
    Define a model of machine learning model as expected by trainer use case.
    """
    def __init__(self, model_trainer,hyperparameter_optimizer=None):
        self.model_trainer = model_trainer
        self.hyperparameter_optimizer = hyperparameter_optimizer
        self.model = None

    def optimize_hyperparameters(self, X_train, y_train):
        if self.hyperparameter_optimizer:
            return self.hyperparameter_optimizer.optimize_hyperparameters(X_train, y_train)
        else:
            return None

    def train(self, X_train, y_train, bestparams:dict):
        self.model = self.model_trainer.train(X_train, y_train, bestparams)
        return self.model

    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

