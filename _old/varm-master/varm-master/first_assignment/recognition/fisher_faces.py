from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from first_assignment.recognition.recognition import Recognition
import numpy as np


class FisherFaces(Recognition):

    def __init__(self, data: np.ndarray, size: tuple, n_examples: int, labels: list, **kwargs):
        print("Recognition Method: {}".format(self.__class__))
        self.size = size
        self.labels = labels
        self.variance = kwargs.get("variance", None)
        # self.data = data
        # self.classes = [[j + i * n_examples for j in range(n_examples)] for i in range(n_classes)]
        self.X = data.T
        self.y = np.array([i for i in range(len(labels)) for j in range(n_examples)])
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42, shuffle=True, stratify=self.y)
        self.linear_discriminant_analysis()
        self.random_forest()
        self.stats()

    def linear_discriminant_analysis(self):
        print("Train Set: {}".format(self.Xtrain.shape))
        print("Test Set: {}".format(self.Xtest.shape))
        self.total_components = self.Xtrain.shape[1]

        self.Xtrain_mu = np.mean(self.Xtrain, axis=0)
        self.Xtrain_n = np.linalg.norm(self.Xtrain, axis=0)
        self.Xtrain_std = np.std(self.Xtrain, axis=0)
        self.Xtest_mu = np.mean(self.Xtest, axis=0)
        self.Xtest_n = np.linalg.norm(self.Xtest, axis=0)
        self.Xtest_std = np.std(self.Xtest, axis=0)

        self.Xtrain = self.Xtrain - self.Xtrain_mu
        self.Xtrain = self.Xtrain / self.Xtrain_std
        self.Xtest = self.Xtest - self.Xtest_mu
        self.Xtest = self.Xtest / self.Xtest_std

        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(self.Xtrain, self.ytrain)
        self.Xtrain = self.lda.transform(self.Xtrain)
        self.Xtest = self.lda.transform(self.Xtest)

    def random_forest(self):
        self.classifier = RandomForestClassifier(max_depth=100, n_estimators=500).fit(self.Xtrain, self.ytrain)
        self.ydec = self.classifier.predict_proba(self.Xtest)[:, 1]
        self.ypred = self.classifier.predict(self.Xtest)

    def stats(self):
        self.cm = confusion_matrix(self.ytest, self.ypred)
        self.recall = recall_score(self.ytest, self.ypred, average='weighted')
        self.precision = precision_score(self.ytest, self.ypred, average='weighted')
        self.accuracy = accuracy_score(self.ytest, self.ypred)
        self.fscore = 2 * self.recall * self.precision / (self.recall + self.precision)
        print("Confusion Matrix: \n{}".format(self.cm))
        print("Accuracy: {:.2f} %".format(round(100 * self.accuracy, 2)))
        print("Recall: {:.2f} %".format(100 * self.recall))
        print("Precision: {:.2f} %".format(100 * self.precision))
        print("F-Score: {:.2f} %".format(100 * self.fscore))
        print(classification_report(self.ytest, self.ypred, target_names=self.labels))

    def classify(self, image: np.ndarray):
        image = image[:, np.newaxis].T
        image = image - self.Xtrain_mu
        image = image / self.Xtrain_std

        image = self.lda.transform(image)
        result = self.classifier.predict(image)
        name = self.labels[result[0]]
        return name
