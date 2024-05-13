import os
import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt


class Registration:

    MARKERS = "./second_assignment/registration/markers/markers_1.jpg"
    MARKERS_SAVE = "./second_assignment/registration/markers/markers.pdf"

    def __init__(self):
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.target_path = os.getcwd() + self.MARKERS_SAVE
        self.example_path = os.getcwd() + self.MARKERS
        assert os.path.isdir(os.path.dirname(self.target_path)), "No directory found at {}.".format(self.target_path)
        assert os.path.isfile(self.example_path), "No file found at {}.".format(self.example_path)
        # self.setup_markers()
        self.load_markers()

    def setup_markers(self):
        fig = plt.figure()
        nx = 4
        ny = 3
        for i in range(1, nx * ny + 1):
            ax = fig.add_subplot(ny, nx, i)
            img = aruco.drawMarker(self.aruco_dict, i, 700)
            plt.imshow(img, cmap='gray', interpolation="nearest")
            ax.axis("off")
        plt.savefig(self.target_path)
        plt.show()

    def load_markers(self, image=None):
        if image is None:
            image = cv2.imread(self.example_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        return frame_markers, corners, ids, rejected

    @staticmethod
    def register(image, virtual_object, marker_corner, marker_id):
        c = marker_corner[0]
        tl = c[0][0], c[0][1]
        tr = c[1][0], c[1][1]
        br = c[2][0], c[2][1]
        bl = c[3][0], c[3][1]
        h, w, c = virtual_object.shape
        points1 = np.array([tl, tr, br, bl])
        points2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix, _ = cv2.findHomography(points2, points1)
        result = cv2.warpPerspective(virtual_object, matrix, (image.shape[1], image.shape[0]))
        cv2.fillConvexPoly(image, points1.astype(int), (0, 0, 0))
        result = image + result
        return result

    @staticmethod
    def plot_markers(frame_markers, corners, ids):
        plt.close('all')
        plt.figure(figsize=(10, 5))
        plt.imshow(frame_markers)
        for i in range(len(ids)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
        plt.legend()
        plt.show()