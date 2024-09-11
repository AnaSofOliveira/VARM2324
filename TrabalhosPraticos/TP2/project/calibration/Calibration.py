import cv2
import numpy as np
from typing import Final
from db.config.Configs import CHESSBOARD_ROWS, CHESSBOARD_COLS, CHESSBOARD_SQUARE_SIZE
from db.Database import Database

class Calibration: 

    stop_criteria: Final = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self) -> None:
        self.__images = Database.load_calibration_images()
        self.__chessboardSize = (CHESSBOARD_COLS, CHESSBOARD_ROWS)

    def calibrate_camera(self, show_error=False): 

        virtual_points, chess_corners, imgs_size = self.__getChessboardCorners()
        _, camera_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(virtual_points, chess_corners, imgs_size, None, None)

        self.__save_calibration_configs(camera_matrix, dist_coef)

        if show_error:
            print("Camera Calibration:: Reprojection Error >", self.calculate_reprojection_error(virtual_points, chess_corners, camera_matrix, dist_coef, rvecs, tvecs))
    
        return camera_matrix, dist_coef, rvecs, tvecs
    

    def recalibrate_camera(self, show_error=False):
        self.__images = Database.load_calibration_images(type='u')
        return self.calibrate_camera(show_error)
        
    def __getChessboardCorners(self):
                
        # Create virtual points to be assigned to chessboard corners
        virtual_pts = np.zeros((self.__chessboardSize[1] * self.__chessboardSize[0], 3), np.float32)
        virtual_pts[:,:2] = np.mgrid[0:self.__chessboardSize[0], 0:self.__chessboardSize[1]].T.reshape(-1, 2)
        virtual_pts = virtual_pts * CHESSBOARD_SQUARE_SIZE

        object_points = []
        image_points = []

        for image in self.__images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.__chessboardSize, None)

            if ret:
                object_points.append(virtual_pts)

                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.stop_criteria)
                image_points.append(corners)

                marked_image = image.copy()
                cv2.drawChessboardCorners(marked_image, self.__chessboardSize, corners, ret)
                cv2.imshow('Chessboard', marked_image)
                cv2.waitKey(500)

        cv2.destroyAllWindows()
        return object_points, image_points, gray.shape[::-1]
    


    def undistort_images(self, camera_matrix, dist_coef): 
        undistorted = []

        for image in self.__images:
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coef, (w, h), 1, (w, h))
            dst = cv2.undistort(image, camera_matrix, dist_coef, None, new_camera_matrix)
            undistorted.append(dst)

        self.__save_undistorted_images(undistorted)

    def __save_undistorted_images(self, images): 
        Database.save_images(images, type='u')
        
    def calculate_reprojection_error(self, object_points, image_points, camera_matrix, dist_coef, rvecs, tvecs):
        total_error = 0

        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coef)
            error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error

        return np.round(total_error / len(object_points), 3)
    
    def __save_calibration_configs(self, camera_matrix, dist_coef):
        Database.save_calibration_config(camera_matrix, dist_coef)