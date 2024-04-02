import os
import cv2
import numpy as np
from preprocess.normalization import Normalization
from filter.filter import Filter
from recognition.lda import LDA
from recognition.pca import PCA
from utils.utils import Utils

from sklearn.neighbors import KNeighborsClassifier


class FaceDetectionAndRecognitionApp:

    def __init__(self):
        self.__classifier1 = KNeighborsClassifier(n_neighbors=3)
        self.__classifier2 = KNeighborsClassifier(n_neighbors=3)
        self.__classifier3 = KNeighborsClassifier(n_neighbors=3)

        self.__pca = None
        self.__lda = None
        self.fit()


    def fit(self):
        
        print("Loading database images...")

        images, labels = Utils.load_training_images()

        self.__pca = PCA().fit(images, labels)
        self.__lda = LDA().fit(images, labels)

        print("Fitting the classifiers...")
        self.__classifier1 = self.__classifier1.fit(images, labels)
        self.__classifier2 = self.__classifier2.fit(self.__pca.project(images), labels)
        self.__classifier3 = self.__classifier3.fit(self.__lda.project(images), labels)


    def classify(self, image):

        try: 

            image = Utils.flatten_image(image)
            image_pca = self.__pca.project(image)
            image_lda = self.__lda.project(image)

            prediction1 = self.__classifier1.predict(image.T)[0]
            prediction2 = self.__classifier2.predict(image_pca)[0]
            prediction3 = self.__classifier3.predict(image_lda)[0]

            return prediction1, prediction2, prediction3
        
        except:
            return "No Identification", "No Identification", "No Identification"
    
    
    def run(self):

        video = cv2.VideoCapture(0)

        
        while True: 

            ret, frame = video.read()

            normalizer = Normalization(frame)
            
            _, norm_test_image = normalizer.normalize(show_images=False)
            
            if norm_test_image is not None:
                face_coords, eyes_coords = normalizer.get_original_face_and_eyes_coords()

                classif, pca_classif, lda_classif = self.classify(norm_test_image)
                
                person = "KNN: " + str(classif)
                person_pca = "PCA: " + str(pca_classif)
                person_lda = "LDA: " + str(lda_classif)

                print("KNN Classifier: ", classif)
                print("KNN Classifier with PCA: ", pca_classif)
                print("KNN Classifier with LDA: ", lda_classif)
                print("")

                frame_with_filter = Filter().apply_filter(frame, pca_classif, eyes_coords)
                frame = frame_with_filter

                cv2.putText(frame, person, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, person_pca, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, person_lda, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video.release()
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    app = FaceDetectionAndRecognitionApp()
    app.run()


