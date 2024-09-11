
import cv2
from cv2 import aruco
import os
import numpy as np
import matplotlib.pyplot as plt 
from projeto.calibracao.Calibracao import Calibracao


def load_markers():
    raise NotImplementedError("Implementar a função load_markers()")
    

def detect_markers(image, show_axis=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    matrix, dist_coeffs, sucesso = Calibracao.carregar_coeficientes()

    print(matrix, dist_coeffs)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

    if (len(corners) > 0) and show_axis:
        for i in range(0, len(ids)): 
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix, dist_coeffs)
            
            cv2.aruco.drawDetectedMarkers(image, corners)
            cv2.drawFrameAxes(image, matrix, dist_coeffs, rvec, tvec, 0.01)

    cv2.imshow('Detected Markers', image)

    return image, corners, ids, rejectedImgPoints


def getVirtualObject(marker_id):
    print("Marker id: ", marker_id)
    virtual_object_path = os.path.join(os.getcwd(), "TrabalhosPraticos/TP2/spikes/virtual_object.jpg")
    virtual_object = cv2.imread(virtual_object_path)

    return virtual_object


def register(image, marker_corners, marker_id):

    virtual_object = getVirtualObject(marker_id)

    first_marker = marker_corners[0]

    top_left = first_marker[0][0], first_marker[0][1]
    top_right = first_marker[1][0], first_marker[1][1]
    bottom_right = first_marker[2][0], first_marker[2][1]
    bottom_left = first_marker[3][0], first_marker[3][1]

    height, width, _ = virtual_object.shape

    point1 = np.array([top_left, top_right, bottom_right, bottom_left])
    point2 = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(point2, point1)

    result = cv2.warpPerspective(virtual_object, matrix, (image.shape[1], image.shape[0]))

    cv2.fillConvexPoly(image, np.int32([point1]), (0, 0, 0))

    result = cv2.add(image, result)

    return result



def plot_markers(image, corners, ids):
    plt.close('all')
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    print("plot_markers:: corners: ", corners, " ids: ", ids)
    for i in range(len(ids)):
        c = corners[i][0]
        print("plot_markers:: c: ", c, c[:, 0].mean(), c[:, 1].mean())
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if ret:
            frame_markers, corners, ids, rejectedImgPoints = load_markers(frame)

            if len(corners) > 0:
                for corner, marker_id in zip(corners, ids):               
                    frame_markers = register(frame, corner, marker_id)

                plot_markers(frame_markers, corners, ids)
            cv2.imshow('Frame', frame_markers)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()