import logging
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import unittest
from src.utils.constants import file_path


class Test_Pre_processing:

    def __init__(self, dataset):
        self.file_path = dataset
        self.credit_card_data = None
        self.X_features = None
        self.y_target = None

    def test_convert_dataframe(self):
        try:
            readfile = pd.read_csv(self.file_path)
            self.credit_card_data = pd.DataFrame(readfile)
            return "Converted to Dataframe"

        except FileNotFoundError:
            return "Failed to Locate File"

        except AttributeError:
            return "Failed to Convert to Dataframe"

    def test_divide_target_feature(self):
        try:
            self.X_features = self.credit_card_data.drop("defaulter", axis=1)
            self.y_target = self.credit_card_data["defaulter"]

            return "Divided to Feature and Target"

        except:
            return "Failed to Divide to Feature and Target"

    def test_split_data(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X_features, self.y_target, test_size=0.3, random_state=42)

            X_train_scaled, X_test_scaled, msg = self.test_scale_data(X_train, X_test)

            X_train_scaled.columns = self.X_features.columns
            X_test_scaled.columns = self.y_target.columns

            y_train.index = X_train.index
            y_test.index = X_test.index
            return "Splited Data Successfully", msg

        except:
            return "Failed to split data into testing and training portion!"

    def test_scale_data(self, X_train, X_test):
        try:
            scaling = StandardScaler()
            X_train = pd.DataFrame(scaling.fit_transform(X_train))
            X_test = pd.DataFrame(scaling.transform(X_test))
            return X_train, X_test, " and Scaled Successfully"
        except:
            return "Failed to perform scaling!"


class Test(unittest.TestCase):

    def test_for_dataframe(self):
        check_df = Test_Pre_processing(file_path)
        if check_df.test_convert_dataframe() == "Converted to Dataframe":
            print("Passed")

    def test_for_feature_target(self):
        feature_target = Test_Pre_processing(file_path)
        if feature_target.test_divide_target_feature() == "Divided to Feature and Target":
            print("Passed")

    "Splited Data Successfully"
    def test_for_split(self):
        split_check = Test_Pre_processing(file_path)
        if split_check.test_split_data() == "Splited Data Successfully and Scaled Successfully":
            print("Passed")


if __name__ == '__main__':
    unittest.main()
