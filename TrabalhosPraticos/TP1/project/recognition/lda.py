
import numpy as np
from .recognition import Recognition
from .pca import PCA

class LDA(Recognition): 

    def __init__(self):
        self.__train_images_pca = []
        self.__train_labels = []
        self.__test_images_pca = []
        self.__eigenvalues = []
        self.__W_fld = []
        
        self.__pca = PCA()
        

    def fit(self, train_images, train_labels):
        self.__train_labels = train_labels
        self.__pca = self.__pca.fit(train_images, train_labels)
        self.__train_images_pca = self.__pca.project(train_images)
        self.__fit_lda(self.__train_images_pca, train_labels)

        return self


    def __fit_lda(self, train_images_pca, labels):
        [n, d] = train_images_pca.shape
        c = np.unique(labels)
        num_components = len(c) - 1

        meanTotal = train_images_pca.mean(axis=0)

        Sw = np.zeros((d, d), dtype=np.float32)
        Sb = np.zeros((d, d), dtype=np.float32)

        for i in c: 
            Xi = train_images_pca[np.where(labels==i)[0], :]
            meanClass = Xi.mean(axis=0)

            Sw = Sw + np.dot((Xi - meanClass).T, (Xi - meanClass))
            Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))

        eigenvalues, eigenvectorsLDA = np.linalg.eig(np.dot(np.linalg.pinv(Sw), Sb))
        idx = np.argsort(-eigenvalues.real)
        eigenvalues, eigenvectorsLDA = eigenvalues[idx], eigenvectorsLDA[:, idx]

        self.__eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
        self.__W_fld = np.array(eigenvectorsLDA[0:, 0:num_components].real, dtype=np.float32, copy=True)


    def project(self, test_images):
        self.__test_images_pca = self.__pca.project(test_images)
        projection = np.dot(self.__test_images_pca, self.__W_fld)
        return projection
    
