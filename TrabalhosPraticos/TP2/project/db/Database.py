import os
import cv2
import glob
import numpy as np
from pathlib import Path

class Database: 

    @staticmethod
    def load_calibration_images(type=''):
        images = []

        if type == 'u':
            regex = os.path.join(Path(__file__).parent, "imgs", "undistorted", "*.jpg")
            images_found = glob.glob(regex)
        else:
            regex = os.path.join(Path(__file__).parent, "imgs", "*.jpg")
            images_found = glob.glob(regex)

        for name in images_found: 
            image = cv2.imread(name, cv2.IMREAD_COLOR)
            images.append(image)

        return images
    
    @staticmethod
    def save_images(images, type=''):

        if type == 'u': 
            path = os.path.join(Path(__file__).parent, "imgs", "undistorted")

            if not os.path.exists(path):
                os.makedirs(path)

            for i, image in enumerate(images):
                cv2.imwrite(os.path.join(path, "chessboard_" + str(i+1) + ".jpg"), image)

    @staticmethod
    def save_calibration_config(camera_matrix, dist_coef):
        path = os.path.join(Path(__file__).parent, "config", "camera_calibration")
        print("Calibration configs saved in ", path)

        if not os.path.exists(path):
            os.makedirs(path)

        cv_file = cv2.FileStorage(os.path.join(path, "intrinsic_params.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', camera_matrix)
        cv_file.write('D', dist_coef)
        cv_file.release()


    @staticmethod
    def load_calibration_config():
        path = os.path.join(Path(__file__).parent, "config", "camera_calibration")
        cv_file = cv2.FileStorage(os.path.join(path, "intrinsic_params.xml"), cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode('K').mat()
        dist_coef = cv_file.getNode('D').mat()
        cv_file.release()
        return camera_matrix, dist_coef
        
    @staticmethod
    def load_virtual_object(marker_id): 
        path = os.path.join(Path(__file__).parent, "imgs", "virtual_objects", str(marker_id) + ".jpg")
        
        virtual_object = cv2.imread(path)

        return virtual_object