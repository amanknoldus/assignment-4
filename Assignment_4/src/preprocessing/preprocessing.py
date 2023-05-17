import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Pre_processing:

    def __init__(self, dataset):
        self.file_path = dataset

    def convert_dataframe(self):
        """Function to convert csv file to dataframe,
        taking csv file from class instance
        @return: processed data
        @rtype: Series
        """

        try:
            readfile = pd.read_csv(self.file_path)
            credit_card_data = pd.DataFrame(readfile)
            return self.divide_target_feature(credit_card_data)

        except FileNotFoundError:
            return "Unable to fetch the file please check path!"

        except AttributeError:
            return "Failed to convert to dataframe"

    def divide_target_feature(self, credit_card_data):
        """
        Function to divide feature and target from the given dataframe
        @param credit_card_data: credit card dataframe
        @return: series[X_train, y_train, X_test, y_test]
        @rtype: Series
        """
        try:
            X_features = credit_card_data.drop("defaulter", axis=1)
            y_target = credit_card_data["defaulter"]
            return self.balance_data(X_features, y_target)
        except:
            return "Failed to divide target & feature!"

    def balance_data(self, X_features, y_target):
        """
        Function to perform balancing of data
        @param X_features: training features
        @param y_target: target columns
        @return: series[X_train, y_train, X_test, y_test]
        @rtype: Series
        """
        try:
            smote = SMOTE()
            X_os, y_os = smote.fit_resample(X_features, y_target)

        except:
            return "Failed to do data balancing!"

        return self.split_data(X_os, y_os, X_features)

    def split_data(self, X_sampled, y_sampled, X_features):
        """
        Function to perform splitting of data into training and testing set
        @param X_sampled: training features
        @param y_sampled: target column
        @param X_features: original features before sampling
        @return: series[X_train, y_train, X_test, y_test]
        @rtype: Series
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42)

            X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)

            X_train_scaled.columns = X_sampled.columns
            X_test_scaled.columns = X_sampled.columns

            y_train.index = X_train.index
            y_test.index = X_test.index
            return X_train_scaled, X_test_scaled, y_train, y_test

        except:
            return "Failed to split data into testing and training portion!"

    @staticmethod
    def scale_data(X_train, X_test):
        """
        Function to perform standardization of data X_train & X_test
        @param X_train:
        @param X_test:
        @return: scaled data(X_train, X_test)
        @rtype: Series
        """
        try:
            scaling = StandardScaler()
            X_train = pd.DataFrame(scaling.fit_transform(X_train))
            X_test = pd.DataFrame(scaling.transform(X_test))
            return X_train, X_test
        except:
            return "Failed to perform scaling!"
