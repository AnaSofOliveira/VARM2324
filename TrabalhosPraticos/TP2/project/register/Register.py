import os
from pathlib import Path
import cv2
import numpy as np
from typing import Final
from db.Database import Database
from db.config.Configs import ARUCO_MARKER_DICT, MAX_MARKER_ID, ARUCO_MARKER_SIZE
from calibration.Calibration import Calibration
import matplotlib.pyplot as plt


class Register: 

    __aruco_dict: Final = cv2.aruco.getPredefinedDictionary(ARUCO_MARKER_DICT)
    __parameters: Final = cv2.aruco.DetectorParameters()


    def __init__(self, calibration: Calibration) -> None:
        self.__calibration = calibration
        self.__camera_matrix, self.__dist_coef = self.__load_camera_parameters()        

    def __calibrate_camera(self, show_error=False):
        camera_matrix, dist_coef, _, _ = self.__calibration.calibrate_camera(show_error)
        self.__calibration.undistort_images(camera_matrix, dist_coef)
        camera_matrix, dist_coef, _, _ = self.__calibration.recalibrate_camera(show_error)
        print("Camera calibrated!")
        return camera_matrix, dist_coef

    def __load_camera_parameters(self):
        print("Loading camera parameters...")
        camera_matrix, dist_coef = Database.load_calibration_config()
        
        if camera_matrix is None or dist_coef is None:
            print("Camera parameters not found. Calibrating camera...")
            camera_matrix, dist_coef = self.__calibrate_camera(show_error=True)
        
        return camera_matrix, dist_coef
    
    def detect_aruco_markers(self, image, show_axis=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.__aruco_dict, parameters=self.__parameters)
        frame = image.copy()
        if (len(corners) > 0) and show_axis:
            
            for i in range(0, len(ids)): 
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.__camera_matrix, self.__dist_coef)
                
                
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self.__camera_matrix, self.__dist_coef, rvec, tvec, 0.01)

        cv2.imshow('Detected Markers', frame)

        return image, corners, ids, rejectedImgPoints
    
    def register_images_on_markers(self, image, markers, ids):
        for marker, marker_id in zip(markers, ids):               
            image = self.__register_image_on_marker(image, marker, marker_id)
        return image
    
    def __register_image_on_marker(self, image, marker, marker_id): 

        try:
            virtual_image = Database.load_virtual_object(marker_id)

            chromakey_backgound_color = np.array([101, 236, 192]) # [B, G, R]

            valor_inferior = chromakey_backgound_color - 20
            valor_superior = chromakey_backgound_color + 20

            virt_img_mask = cv2.inRange(virtual_image, valor_inferior, valor_superior)
            virt_img_mask_inv = 255 - virt_img_mask
            offset = 50
            
            top_left = marker[0][0] - offset, marker[0][1] - offset
            top_right = marker[1][0] + offset, marker[1][1] - offset
            bottom_right = marker[2][0] + offset, marker[2][1] + offset
            bottom_left = marker[3][0] - offset, marker[3][1] + offset

            height, width, _ = virtual_image.shape

            point1 = np.array([top_left, top_right, bottom_right, bottom_left])
            point2 = np.array([[0, 0], [width, 0], [width, height], [0, height]])

            matrix, _ = cv2.findHomography(point2, point1)

            result_virtual_image = cv2.warpPerspective(virtual_image, matrix, (image.shape[1], image.shape[0]))
            result_mask = cv2.warpPerspective(virt_img_mask_inv, matrix, (image.shape[1], image.shape[0]))
            
            all_img_mask_inv = 255-result_mask

            virtual_image_no_back = cv2.bitwise_and(result_virtual_image,result_virtual_image,mask = result_mask)

            image_no_back = cv2.bitwise_and(image,image,mask = all_img_mask_inv)

            result = cv2.add(image_no_back, virtual_image_no_back)

            return result
        
        except Exception as e:
            print("Unable to register image: ", e)
            return image
    
    @staticmethod
    def generate_aruco_markers(): 
        for i in range(MAX_MARKER_ID): 
            marker = cv2.aruco.generateImageMarker(Register.__aruco_dict, i, ARUCO_MARKER_SIZE)
            path = os.path.join(Path(__file__).parent.parent, "db", "imgs", "markers")

            if not os.path.exists(path):
                os.makedirs(path)

            cv2.imwrite(os.path.join(path, f"marker_{i}.jpg"), marker)
        print("Generated markers available in", Path(path))