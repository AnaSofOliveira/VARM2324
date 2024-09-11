import numpy as np
import cv2 as cv
import glob

from projeto.definicoes.Definicoes import *


def calibrate():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    nrows = CALIBRATION_WIDTH
    ncols = CALIBRATION_HEIGHT

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ncols*nrows,3), np.float32)
    objp[:,:2] = np.mgrid[0:nrows,0:ncols].T.reshape(-1,2)
    objp = objp * CALIBRATION_SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    print(CALIBRATION_REGEX)
    images = glob.glob(CALIBRATION_REGEX)

    print(images)
    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (nrows,ncols), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("valid\n")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (nrows,ncols), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    
    #cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("\n")
    print("Dist : \n")
    print(dist)
    print("\n")
    print("rvecs : \n")
    print(rvecs)
    print("\n")
    print("tvecs : \n")
    print(tvecs)
    print("\n")

    print(CALIBRATION_CAM_SAVE_CONFIGS)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    
    cv_file = cv.FileStorage(CALIBRATION_CAM_SAVE_CONFIGS, cv.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

