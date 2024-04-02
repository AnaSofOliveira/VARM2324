import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

path = 'C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP1\\database_images\\normalized\\all'

print(os.listdir(path))

X = None
y = None
for filename in os.listdir(path): 
    classe = filename.split(".")[0][:-1]
    
    # Lê imagem
    image = cv2.imread(os.path.join(path, filename))

    # Converter para cinzento
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flatten a imagem
    flat_image = gray_image.flatten()[:, np.newaxis]

    # Junta imagem à matriz
    if X is None: 
        X = flat_image.T
        y = classe
    else: 
        X = np.vstack((X, flat_image.T))
        y = np.hstack((y, classe))


print(X.shape)
print(y.shape)

shuffled_indices = np.random.permutation(X.shape[0])
X = X[shuffled_indices]
y = y[shuffled_indices]

spitIndex = 4

X1 = X[:spitIndex]
y1 = y[:spitIndex]

X2 = X[spitIndex:]
y2 = y[spitIndex:]

print("y1: ", y1)
print("y2: ", y2)

plt.plot(y1, alpha=0.5)
plt.plot(y2, alpha=0.5)
plt.show()

print("Shape X1: ", X1.shape)
print("Shape X2: ", X2.shape)


'''
############################################################################################################
##########################                      EIGENFACES                       ###########################
############################################################################################################
'''

[n, d] = X1.shape
num_components = n

mu = X1.mean(axis=0)
X1 = X1 - mu

if n > d: 
    C = np.dot(X1.T, X1)
    [eigenvalues, eigenvectors_pca] = np.linalg.eig(C)

else:
    C = np.dot(X1, X1.T)
    [eigenvalues, eigenvectors_pca] = np.linalg.eig(C)
    eigenvectors_pca = np.dot(X1.T, eigenvectors_pca)

    for i in range(n):
        eigenvectors_pca[:, i] = eigenvectors_pca[:, i] / np.linalg.norm(eigenvectors_pca[:, i])

idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors_pca = eigenvectors_pca[:, idx]

eigenvalues = eigenvalues[0:num_components].copy()
eigenvectors_pca = eigenvectors_pca[:, 0:num_components].copy()


def project(eigenvectors_pca, X, mu_pca):
    print()
    print("Project")
    print("X Shape: ", X.shape)
    print("mu Shape: ", mu_pca.shape)
    print("eigenvectors Shape: ", eigenvectors_pca.shape) 
    print()
    return np.dot(X - mu_pca, eigenvectors_pca)


'''
# Mostrar eigenfaces
eigenfaces = []
for i in range(X1.shape[0]): 
    eigenface = eigenvectors[:,i].reshape(56, 46)
    eigenfaces.append(eigenface)
    plt.figure()
    plt.title('Eigenface ' + str(i))
    plt.imshow(eigenface, cmap='gray')
    plt.show()'''

X1pca = project(eigenvectors_pca, X1, mu)
print("X1pca Shape: ", X1pca.shape, "\n", X1pca)

'''
############################################################################################################
##########################                      FISHERFACES                       ##########################
############################################################################################################
'''

[n, d] = X1pca.shape
c = np.unique(y1)
num_components = len(c) - 1

meanTotal = X1pca.mean(axis=0)

Sw = np.zeros((d, d), dtype=np.float32)
Sb = np.zeros((d, d), dtype=np.float32)

for i in c: 
    Xi = X1pca[np.where(y1==i)[0], :]
    meanClass = Xi.mean(axis=0)

    Sw = Sw + np.dot((Xi - meanClass).T, (Xi - meanClass))
    Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))


eigenvalues, eigenvectorsLDA = np.linalg.eig(np.dot(np.linalg.pinv(Sw), Sb))
idx = np.argsort(-eigenvalues.real)
eigenvalues, eigenvectorsLDA = eigenvalues[idx], eigenvectorsLDA[:, idx]
eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
eigenvectorsLDA = np.array(eigenvectorsLDA[0:, 0:num_components].real, dtype=np.float32, copy=True)

X1lda = np.dot(X1pca, eigenvectorsLDA)

print("X1lda Shape: ", X1lda.shape, "\n", X1lda)


'''
############################################################################################################
################                         CLASSIFICATION USING PCA                           ################
############################################################################################################
'''

KNNpca = KNeighborsClassifier(n_neighbors=3).fit(X1pca, y1)
print("X2 Shape: ", X2.shape)
print("my Shape: ", mu.shape)
print("eigenvectors Shape: ", eigenvectors_pca.shape)

X2pca = project(eigenvectors_pca, X2, mu)

print("X1pca Shape: ", X1pca.shape)
print("X2pca Shape: ", X2pca.shape)
pred = KNNpca.predict(X2pca)

print("pred PCA: ", pred)
print("Accuracy PCA: ", np.sum(pred == y2))
print(confusion_matrix(y2, pred))


'''
############################################################################################################
################                         CLASSIFICATION USING LDA                           ################
############################################################################################################
'''
KNNlda = KNeighborsClassifier(n_neighbors=3).fit(X1lda, y1)
X2lda = np.dot(X2pca, eigenvectorsLDA)

print("X1lda Shape: ", X1lda.shape)
print("X2lda Shape: ", X2lda.shape)

pred = KNNlda.predict(X2lda)

print("pred LDA: ", pred)
print("Accuracy LDA: ", np.sum(pred == y2))
print(confusion_matrix(y2, pred))

