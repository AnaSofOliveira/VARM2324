import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from first_assignment.recognition.recognition import Recognition


class EigenFaces(Recognition):

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
        self.principal_component_analysis()
        self.random_forest()
        self.stats()

    def principal_component_analysis(self):
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

        self.pca = PCA(n_components=self.variance)
        self.pca.fit(self.Xtrain)
        self.Xtrain = self.pca.transform(self.Xtrain)
        self.Xtest = self.pca.transform(self.Xtest)
        self.k_components = sum(max(np.argwhere(np.cumsum(self.pca.explained_variance_ratio_) > self.variance)))
        if len(self.size) <= 2:
            self.eigenfaces = self.pca.components_.reshape((self.k_components + 1, self.size[0], self.size[1]))
        else:
            self.eigenfaces = self.pca.components_.reshape((self.k_components + 1, self.size[0], self.size[1], self.size[2]))

        # n_samples = self.Xtrain.shape[0]
        # cov_matrix = np.dot(self.Xtrain_mu.T, self.Xtrain_mu) / n_samples
        # eigenvalues = self.pca.explained_variance_
        #
        # for eigenvalue, eigenvector in zip(eigenvalues, self.pca.components_):
        #     print(eigenvector)
        #     print(eigenvalue)
        #
        # w = np.dot(self.pca.components_.T, self.Xtrain.T)
        # w = w / np.linalg.norm(w, axis=0)
        # print(w.shape)
        # print(np.round(np.dot(w.T, w)))

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

        plt.figure(figsize=(8, 4))
        plt.yscale('log')
        plt.title('Principal Component Analysis ({} of {} Features for {} Variance) '.format(
            self.k_components, self.total_components, self.variance))
        plt.xlabel('Number of Features')
        plt.ylabel('Variance (Logarithmic)')
        plt.plot(self.pca.explained_variance_ratio_)
        plt.show()

        if len(self.eigenfaces) < 12:
            return
        plt.figure(figsize=(1.4 * 3, 1.2 * 4))
        plt.suptitle('Most Relevant (12) Eigenfaces in Subspace')
        for i in range(3 * 4):
            plt.subplot(3, 4, i + 1)
            if len(self.eigenfaces[0].shape) > 2:
                plt.imshow(self.eigenfaces[i][:, :, 0].reshape((self.size[0], self.size[1])), cmap=plt.cm.gray)
            else:
                plt.imshow(self.eigenfaces[i].reshape((self.size[0], self.size[1])), cmap=plt.cm.gray)
            plt.title(i, size=8)
            plt.xticks(())
            plt.yticks(())
        plt.show()

    def classify(self, image: np.ndarray):
        image = image[:, np.newaxis].T
        image = image - self.Xtrain_mu
        image = image / self.Xtrain_std

        image = self.pca.transform(image)
        result = self.classifier.predict(image)
        name = self.labels[result[0]]
        return name

    # ------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------First Version -----------------------------------------------------
    # def principal_component_analysis(self):
    #     a = self.data
    #     n = self.n_components
    #     print("N Components: {}".format(n))
    #
    #     # Determine the mean face m (size of 1xn)
    #     m = a.mean(axis=1)[:, np.newaxis].T
    #     print("M: {}".format(m.shape))
    #
    #     # Define matrix A (size of n×N) whose columns contain the "AC components" of the faces from the training set
    #     a -= m.T
    #     print("A: {}".format(a.shape))
    #
    #     # Compute the eigenvectors and eigenvalues of matrix R = ATA (size of N×N)
    #     r = np.dot(a.T, a)
    #     print("R: {}".format(r.shape))
    #     eigen_values, eigen_vectors = np.linalg.eig(r)
    #
    #     # Select m (maximum of N - 1) eigenvectors from R, associated to the highest eigenvalues.
    #     index = (-eigen_values).argsort()[:n-1]
    #     print("M Eigen Vectors: {}".format(index))
    #
    #     # Define matrix V (N×m), formed by the m eigenvectors of R
    #     v = eigen_vectors[:, index]
    #     print("Matrix V: {}".format(v.shape))
    #
    #     # Get matrix W (n×m) by the relation W = AV (not forgetting that the W columns form an orthonormal basis)
    #     w = np.dot(a, v)
    #     w = w / np.linalg.norm(w, axis=0)
    #     print("Matrix W: {}".format(w.shape))
    #
    #     # np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, threshold=np.inf)
    #     # t = np.dot(w.T, w)
    #     # i = np.eye(t.shape[0]).astype(np.uint8)
    #     # assert (t.shape[0] == t.shape[1]) and (i == t).all(), "Orthonormal basis of W implies dot product of W.T and W should equal identity matrix."
    #
    #     self.m = m
    #     self.w = w
    #
    # def nearest_neighbour(self, image: np.ndarray):
    #     test_img = image.flatten()
    #     test_normalized = (test_img - self.m.T)
    #     test_weight = test_normalized.dot(self.w)
    #     result = np.argmin(np.linalg.norm(test_weight - self.w, axis=0))
    #     print(result)
    #     for index, subset in enumerate(self.classes):
    #         if result in subset:
    #             print(self.labels[index])
    #             break
    #
    # def stats(self):
    #     try:
    #         cv2.imshow("Result", self.mu.reshape(self.size).astype(np.uint8))
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #         print("".format(self))
    #         print("N Components: {}".format(self.n_components))
    #         print("Average Face: {}".format(self.average_face.shape))
    #         print("Data: {}".format(self.data.shape))
    #         print("Covariance Matrix: {}".format(self.covariance_matrix.shape))
    #         print("Eigen Values: {}".format(self.eigen_values.shape))
    #         print("Eigen Vectors: {}".format(self.eigen_vectors.shape))
    #         print("K Eigen Vectors: {}".format(self.k_eigen_vectors.shape))
    #         print("Eigen Faces: {}".format(self.eigen_faces.shape))
    #         print("Weights: {}".format(self.weights.shape))
    #         print("Weights: {}".format(self.weights))
    #     except AttributeError as e:
    #         print("Object attribute has not been defined.")

    # ------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------Second Version ----------------------------------------------------
    # def principal_component_analysis(self):
    #     self.average_face = self.data.mean(axis=1)
    #     self.data -= self.average_face[:, np.newaxis]
    #     self.covariance_matrix = np.cov(self.data.T)
    #     self.eigen_values, self.eigen_vectors = np.linalg.eig(self.covariance_matrix)
    #     self.k_eigen_vectors = self.eigen_vectors[np.argsort(self.eigen_values).flatten()][:self.n_components]
    #     self.eigen_faces = self.k_eigen_vectors.dot(self.data.T)
    #     self.weights = self.data.T.dot(self.eigen_faces.T)
    #
    # def nearest_neighbour(self, image: np.ndarray):
    #     test_img = image.flatten()
    #     test_normalized = test_img - self.average_face
    #     test_weight = test_normalized.T.dot(self.eigen_faces.T)
    #     result = np.argmin(np.linalg.norm(test_weight - self.weights, axis=1))
    #     for index, subset in enumerate(self.classes):
    #         if result in subset:
    #             print(self.labels[index])
    #             break
    #
    # def stats(self):
    #     try:
    #         cv2.imshow("Result", self.average_face.reshape(self.size).astype(np.uint8))
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #         print("".format(self))
    #         print("N Components: {}".format(self.n_components))
    #         print("Average Face: {}".format(self.average_face.shape))
    #         print("Data: {}".format(self.data.shape))
    #         print("Covariance Matrix: {}".format(self.covariance_matrix.shape))
    #         print("Eigen Values: {}".format(self.eigen_values.shape))
    #         print("Eigen Vectors: {}".format(self.eigen_vectors.shape))
    #         print("K Eigen Vectors: {}".format(self.k_eigen_vectors.shape))
    #         print("Eigen Faces: {}".format(self.eigen_faces.shape))
    #         print("Weights: {}".format(self.weights.shape))
    #     except AttributeError as e:
    #         print("Object attribute has not been defined.")
