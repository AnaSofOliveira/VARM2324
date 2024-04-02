import numpy as np
from .recognition import Recognition

class PCA(Recognition): 

    def __init__(self):
        self.__num_components = 0
        self.__train_images = []
        self.__train_labels = []
        self.__eigenvalues = []
        self.__W = []


    def fit(self, train_images, train_labels=None):

        self.__train_images = train_images
        self.__train_labels = train_labels

        [n, d] = self.__train_images.shape
        self.__num_components = n - 1

        self.__train_images_mean = self.__train_images.mean(axis=0)
        self.__centered_train_images = self.__train_images - self.__train_images_mean

        if n > d: 
            C = np.dot(self.__centered_train_images.T, self.__centered_train_images)
            [eigenvalues, eigenvectors_pca] = np.linalg.eig(C)

        else:
            C = np.dot(self.__centered_train_images, self.__centered_train_images.T)
            [eigenvalues, eigenvectors_pca] = np.linalg.eig(C)
            eigenvectors_pca = np.dot(self.__centered_train_images.T, eigenvectors_pca)

            for i in range(n):
                eigenvectors_pca[:, i] = eigenvectors_pca[:, i] / np.linalg.norm(eigenvectors_pca[:, i])

        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors_pca = eigenvectors_pca[:, idx]

        self.__eigenvalues = eigenvalues[0:self.__num_components].copy()
        self.__W = eigenvectors_pca[:, 0:self.__num_components].copy()

        return self

    def project(self, test_images):
        projection = np.dot(test_images - self.__train_images_mean, self.__W)
        return projection

