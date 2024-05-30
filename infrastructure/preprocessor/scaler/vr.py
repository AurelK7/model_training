from application.ports.ports import PreprocessorInterface
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import date, datetime

import logging

logger = logging.getLogger(__name__)

class DataPreprocessor(PreprocessorInterface):

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
        # Convert mec to car age
        def convertAgeCar(date_in):
            # convert date in datetime
            day, month, year = [int(elt) for elt in date_in.split('/')]
            #from datetime import date
            dat = date(year, month, day)
            # convert to age
            age = 12 * (datetime.now().date().year - dat.year) + (
                        datetime.now().date().month - dat.month)
            return dat.year, age

        data.insert(3, 'annee', data['miseEnCirculation'].apply(lambda x:
                                                                convertAgeCar(x)[0]))
        data.insert(4, 'carAge', data['miseEnCirculation'].apply(lambda x:
                                                                convertAgeCar(x)[1]))
        data = data.drop(columns='miseEnCirculation')

        # reformatage des features transmission & carburant
        data['transmission'] = data.transmission.replace({'Robotisée': 'Automatique'})
        data['carburant'] = data.carburant.replace({'HybrideDieselRechargeable': 'Hybride'})
        data['kilometrage'] = data.kilometrage.apply(lambda x: str(x).split(' km')[0])
        data['emissionCo2'] = data['emissionCo2'].apply(lambda x: str(x).split('g')[0])
        # Change data types
        data = data.astype({
            'marque': 'category',
            'modele': 'category',
            'version': 'category',
            'carrosserie': 'category',
            'carburant': 'category',
            'transmission': 'category',
            'couleur': 'category',
            'kilometrage': 'int',
            'emissionCo2': 'int'

        })
        # Drop columns : ['mensualite','version','couleur','places','portes','carrosserie','vitesses','prix']
        #data = data[data.mensualite == 60]
        cols = ['mensualite','version','couleur','places','portes','carrosserie',
                'vitesses', 'puissanceCv', 'prix']
        #cols = []
        if 'prix' not in cols:
            data = data.drop(index=data[data.prix == -1].index)
        data = data.drop(columns=cols)
        #print(data.describe())
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
        num_cols = [col for col in X_train.columns if
         X_train[col].dtype not in ['object', 'category']]
        X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])
        return X_train, X_test

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
        X_train, X_test = self.scale_data(X_train, X_test)
        # replace numerical features with scaled features
        return X_train, X_test, y_train, y_test

    def get_scaler(self):
        """Get the scaler."""
        return self.scaler