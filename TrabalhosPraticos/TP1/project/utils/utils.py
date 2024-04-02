import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Utils:

    @staticmethod
    def show_train_test_distribution(y1, y2): 
        plt.plot(y1, alpha=0.5)
        plt.plot(y2, alpha=0.5)
        plt.show()


    @staticmethod
    def read_image(image_path, style='rgb'):
        if style == 'gray':
            return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

        if style == 'rgb':
            return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        return cv2.imread(image_path)

    @staticmethod
    def flatten_image(image): 
        return image.flatten()[:, np.newaxis]
    
    @staticmethod
    def load_training_images(): 

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\database\\normalized\\")

        images = None
        labels = None
        for filename in os.listdir(path): 
            image_path = os.path.join(path, filename)
            gray_image = Utils.read_image(image_path, style='gray')
            flat_image = Utils.flatten_image(gray_image)

            classe = filename.split(".")[0][:-1]

            if images is None: 
                images = flat_image.T
                labels = classe
            else: 
                images = np.vstack((images, flat_image.T))
                labels = np.hstack((labels, classe))

        return images, labels