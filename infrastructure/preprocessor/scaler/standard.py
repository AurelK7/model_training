from application.ports.ports import PreprocessorInterface
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)

class StandardDataPreprocessor(PreprocessorInterface):
    """
    Sklearn StandardScaler implementation of ports.PreprocessorInterface.
    """

    def __init__(self, y_label, test_size: float=0.2, random_state: int = 42) -> None:
        """Instantiate the data preprocessor and scaler with sklearn StandardScaler. """
        self.y_label = y_label
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()


    def clean_data(self, data):
        """
        Clean the dataset.
        :param data: dataset to clean
        :return: cleaned dataset
        """
        logger.debug("Cleaning the dataset...")
        # TODO: Add data cleaning steps here if needed
        return data

    def split_data(self, X, y):
        """
        Split the dataset into train and test set.
        """
        logger.debug("Spliting dataset into train test with test_size:...")
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def scale_data(self, X_train, X_test):
        """
        Scale the numerical features.
        """
        logger.debug("Scaling X_train and X_test...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def format(self, data):
        """
        Format the dataset. Split the dataset into train and test set and scale numerical features.
        """
        logger.debug(f"Formating the dataset with ylabel={self.y_label}, test_size={self.test_size} and random_state"
                     f"={self.random_state}...")
        data_cleaned = self.clean_data(data)
        y = data_cleaned[self.y_label]
        X = data_cleaned.drop(self.y_label, axis=1)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        # scale numerical features
        colnum = [col for col in X_train.columns if X_train[col].dtype not in ['object','category']]
        X_train_num, X_test_num = self.scale_data(X_train[colnum], X_test[colnum])
        # replace numerical features with scaled features
        allcols = X_train.columns
        X_train = X_train.drop(colnum, axis=1)
        X_test = X_test.drop(colnum, axis=1)
        for index, col in enumerate(allcols):
            if col in colnum:
                X_train.insert(index, col, X_train_num[:, index])
                X_test.insert(index, col, X_test_num[:, index])

        return X_train, X_test, y_train, y_test

    def get_scaler(self):
        """Get the scaler."""
        return self.scaler