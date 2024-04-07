import os 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from classification.classification import Classification
from recognition.lda import LDA
from recognition.pca import PCA
from utils.utils import Utils


project_path = os.path.dirname(os.path.abspath(__file__))
norm_path = os.path.join(project_path, "database\\normalized\\")

print(os.listdir(norm_path))

X = None
y = None
for filename in os.listdir(norm_path): 
    classe = filename.split(".")[0][:-1]

    image_path = os.path.join(norm_path, filename)
    
    gray_image = Utils.read_image(image_path, style='gray')
    
    # Flatten a imagem
    flat_image = Utils.flatten_image(gray_image)

    # Junta imagem à matriz
    if X is None: 
        X = flat_image.T
        y = classe
    else: 
        X = np.vstack((X, flat_image.T))
        y = np.hstack((y, classe))


print(X.shape)
print(y.shape)

np.random.seed(42)
shuffled_indices = np.random.permutation(X.shape[0])
X = X[shuffled_indices]
y = y[shuffled_indices]

spitIndex = y.shape[0] // 2
# Imagens de Treino
X1 = X[:spitIndex]
y1 = y[:spitIndex]

# Imagens de Teste
X2 = X[spitIndex:]
y2 = y[spitIndex:]

print("y1: ", y1)
print("y2: ", y2)

Utils.show_train_test_distribution(y1, y2)

print("Shape X1: ", X1.shape)
print("Shape X2: ", X2.shape)


model1 = Classification(X1, y1, X2, y2)
model1.fit()
model1_pred = model1.predict()
model1_score = model1.evaluate()
model1_roc = model1.roc_curve()


############################################################################################################
##########################                      EIGENFACES                       ###########################
############################################################################################################
n_components = X.shape[0]

pca = PCA().fit(X1)
X1_pca = pca.project(X1)
X2_pca = pca.project(X2)

model2 = Classification(X1_pca, y1, X2_pca, y2)
model2.fit()
model2_pred = model2.predict()
model2_score = model2.evaluate()
model2_roc = model2.roc_curve()


############################################################################################################
##########################                      FISHERFACES                      ###########################
############################################################################################################

lda = LDA().fit(X1, y1)
X1_lda = lda.project(X1)
X2_lda = lda.project(X2)

model3 = Classification(X1_lda, y1, X2_lda, y2)
model3.fit()
model3_pred = model3.predict()
model3_score = model3.evaluate()
model3_roc = model3.roc_curve()


############################################################################################################
#######################                      MODELS EVALUATION                      ########################
############################################################################################################


print("Model 1 Test Score: ", model1_score)
print("Model 2 Test Score: ", model2_score)
print("Model 3 Test Score: ", model3_score)


'''# On each teste image, plot the image with the predicted label and the true label
for i in range(X2.shape[0]):
    
    plt.subplots(1, 3, figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(X2[i].reshape(56, 46), cmap='gray')
    plt.title(f"Predicted: {model1_pred[i]} - True: {y2[i]}")

    plt.subplot(1, 3, 2)    
    plt.imshow(X2[i].reshape(56, 46), cmap='gray')
    plt.title(f"Predicted: {model2_pred[i]} - True: {y2[i]}")
    
    plt.subplot(1, 3, 3)
    plt.imshow(X2[i].reshape(56, 46), cmap='gray')
    plt.title(f"Predicted: {model3_pred[i]} - True: {y2[i]}")
    plt.show()

'''

# Plot the confusion matrix for each model
print("Classes: ", np.unique(y2))
print("Model 1 Confusion Matrix")
print(confusion_matrix(y2, model1_pred))

print("Model 2 Confusion Matrix")
print(confusion_matrix(y2, model2_pred))

print("Model 3 Confusion Matrix")
print(confusion_matrix(y2, model3_pred))



