import numpy as np
import cv2 as cv
import glob

chessboadSize = (6, 7)
#frameSize = (640, 480)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboadSize[0] * chessboadSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboadSize[0], 0:chessboadSize[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP2\\spikes\\images\\camera1\\*.jpg")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboadSize, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        print("objeto identificado: ", objp)
        print("objeto na imagem: ", corners2)

        cv.drawChessboardCorners(img, chessboadSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(10)


print(len(objpoints), "\n", len(imgpoints))
print(len(objpoints[0]), "\n", len(imgpoints[0]))
print(len(objpoints[3]), "\n", len(imgpoints[3]))
print(len(objpoints[0][0]), "\n", len(imgpoints[0][0]))
########## Calibrate camera ##########

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : ", mtx)
print("\nDist : ", dist)
print("\nrvecs : ", rvecs)
print("\ntvecs : ", tvecs)


########## Undistort ##########

img = cv.imread('C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP2\\spikes\\images\\camera1\\camara_1 (5).jpg')
h, w = img.shape[:2]

newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, mtx, dist, None, newCameraMatrix)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP2\\spikes\\images\\camera1\\undistorted.jpg', dst)

#Undistort with remapping
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP2\\spikes\\images\\camera1\\undistorted_remap.jpg', dst)


# Reprojection error

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total error: ", mean_error / len(objpoints))


########### END CAMERA CALIBRATION ###########


import numpy as np
import cv2 as cv
import glob

#with np.load("Camera.npz") as file: 
#    mtx, dist, _, _ = [file[i] for i in ("mtx", "dist", "rvecs", "tvecs")]

def draw(image, corners, imgpts):
    
    corner = tuple(corners[0].ravel())

    x = imgpts[0].ravel()
    y = imgpts[1].ravel()
    z = imgpts[2].ravel()

    print("Image Shape: ", image.shape, type(image.shape))
    print("Corner: ", corner, type(corner))
    print("Axis x: ", imgpts[0].ravel(), type(imgpts[0].ravel()))
    print("Axis y: ", tuple(imgpts[1].ravel()), type(tuple(imgpts[1].ravel())))
    print("Axis z: ", tuple(imgpts[2].ravel()), type(tuple(imgpts[2].ravel())))

    image = cv.line(image, (int(corner[0]), int(corner[1])), (int(x[0]), int(x[1])), (255, 0, 0), 5)
    image = cv.line(image, (int(corner[0]), int(corner[1])), (int(y[0]), int(y[1])), (0, 255, 0), 5)
    image = cv.line(image, (int(corner[0]), int(corner[1])), (int(z[0]), int(z[1])), (0, 0, 255), 5)

    return image


def drawBoxes(image, corners, imgpts): 
    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv.drawContours(image, [imgpts[:4]], -1, (0, 255, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), -3)

    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
axisBoxes = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])

for fname in glob.glob('C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP2\\spikes\\images\\camera1\\undistorted*.jpg'):
    image = cv.imread(fname)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboadSize, None)

    if ret:

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        #imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)
        
        #image = draw(image, corners2, imgpts)
        image = drawBoxes(image, corners2, imgpts)

        cv.imshow('img', image)
        cv.waitKey(0)
        
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite('C:\\Users\\ana.sofia.oliveira\\Documents\\ISEL\\VARM2324\\TrabalhosPraticos\\TP2\\spikes\\images\\camera1\\pose' + image, image)

cv.destroyAllWindows()
