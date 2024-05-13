import os
import cv2
import numpy as np
import pathlib

from second_assignment.second_database.second_database import SecondDatabase


class Calibration:
    RESOURCES = "second_assignment\\calibration\\resources\\"

    def __init__(self, image_format, square_size, width, height):
        self.image_format = image_format
        self.square_size = square_size
        self.height = height
        self.width = width
        self.examples = SecondDatabase.load('examples').values()
        self.resources = "{}\\{}".format(os.getcwd(), Calibration.RESOURCES)
        assert os.path.isdir(self.resources), "Directory was not found at {}.".format(self.resources)

    def calibrate_chessboard(self):
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
        objp = np.zeros((self.height * self.width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        objp = objp * self.square_size
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        gray = None
        for img in self.examples:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (self.width, self.height), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
        # Calibrate camera
        _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        Calibration.projection_error(objpoints, imgpoints, cameraMatrix, distCoeffs, rvecs, tvecs)
        return _, cameraMatrix, distCoeffs, rvecs, tvecs

    @staticmethod
    def projection_error(objpoints, imgpoints, cameraMatrix, distCoeffs, rvecs, tvecs):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("Calibration :: Total error: {}".format(mean_error / len(objpoints)))

    @staticmethod
    def save_coefficients(cameraMatrix, distCoeffs):
        """Save the camera matrix and the distortion coefficients to given path/file."""
        cv_file = cv2.FileStorage(Calibration.RESOURCES + "calibration", cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', cameraMatrix)
        cv_file.write('D', distCoeffs)
        cv_file.release()

    @staticmethod
    def load_coefficients():
        """Loads camera matrix and distortion coefficients."""
        cv_file = cv2.FileStorage(Calibration.RESOURCES + "calibration", cv2.FILE_STORAGE_READ)
        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        camera_matrix = cv_file.getNode('K').mat()
        dist_matrix = cv_file.getNode('D').mat()
        cv_file.release()
        return camera_matrix, dist_matrix
