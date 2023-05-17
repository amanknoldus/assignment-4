from src.preprocessing.preprocessing import Pre_processing
from src.utils.constants import file_path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import unittest


class Test_SVM_Classification:
    def __init__(self):
        self.dataset = file_path
        self.processed_data = None

    def test_apply_model(self):
        try:
            data = Pre_processing(self.dataset)
            processed_data = data.convert_dataframe()
            return "Pre Processing Done Successfully"
        except:
            return "Failed to do Pre Processing"

    def test_svm_classification(self):
        try:
            X_train = self.processed_data[0]
            X_test = self.processed_data[1]
            y_train = self.processed_data[2]
            y_test = self.processed_data[3]

            model = SVC(kernel='rbf')
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)
            score = accuracy_score(y_test, prediction)
            return "Calculated Accuracy Successfully"
        except:
            return "Some Error Occurred in Process"


class Test(unittest.TestCase):

    def test_for_apply_model(self):
        check_apply_model = Test_SVM_Classification()
        if check_apply_model.test_apply_model() == "Pre Processing Done Successfully":
            print("Passed")

    def test_for_svm_classification(self):
        check_apply_model = Test_SVM_Classification()
        if check_apply_model.test_svm_classification() == "Calculated Accuracy Successfully":
            print("Passed")


if __name__ == '__main__':
    unittest.main()
