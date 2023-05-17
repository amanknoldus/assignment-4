from src.preprocessing.preprocessing import Pre_processing
from src.utils.constants import file_path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVM_Classification:
    def __init__(self):
        self.dataset = file_path

    def apply_model(self):
        """
        Function to create instance of Pre_processing class,
        passing file path for preprocessing,
        then sending that data to svm_classification function.
        """
        data = Pre_processing(self.dataset)
        processed_data = data.convert_dataframe()
        self.svm_classification(processed_data)

    @staticmethod
    def svm_classification(processed_data):
        """
        Function to make prediction using SVM(kernel="rbf"),
        using X_train and X_test, then make prediction
        according to X_test data and then checking accuracy
        using y_test data.
        """
        try:
            X_train = processed_data[0]
            X_test = processed_data[1]
            y_train = processed_data[2]
            y_test = processed_data[3]

            model = SVC(kernel='rbf')
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            score = accuracy_score(y_test, prediction)
            print("Accuracy using SVM is:", score)

        except:
            print("Some Error Occurred in Process")
