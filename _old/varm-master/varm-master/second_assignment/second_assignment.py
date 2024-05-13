import cv2
import time

from second_assignment.second_database.second_database import SecondDatabase


class SecondAssignment:
    RESULT_IMAGE = "./second_assignment/calibration/examples/undistorted_example.jpg"

    def __init__(self, calibration, registration):
        self.registration = registration
        self.calibration = calibration
        self.result_path = "{}{}".format(SecondDatabase.RESULTS, self.RESULT_IMAGE)
        self.setup_calibration()
        self.camera_matrix, self.dist_matrix = self.load_calibration()
        self.objects = SecondDatabase.load('objects', listing=True)

    def setup_calibration(self):
        _, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = self.calibration.calibrate_chessboard()
        # self.cameraMatrix = self.extra_step(self.camera_matrix, self.dist_coeffs)
        self.calibration.save_coefficients(self.camera_matrix, self.dist_coeffs)

    def extra_step(self):
        example = cv2.imread(self.result_path)
        h, w = example.shape[:2]
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h))
        # cv2.imshow('example', roi)
        # cv2.waitKey(0)
        return new_matrix

    def load_calibration(self):
        return self.calibration.load_coefficients()

    def undistort(self, image):
        result = cv2.undistort(image, self.camera_matrix, self.dist_matrix, None, None)
        cv2.imwrite(self.result_path, result)
        return result

    def detect_markers(self, image):
        frame_markers, corners, ids, _ = self.registration.load_markers(image)
        status = True if ids is not None and len(corners) > 0 else False
        return frame_markers, corners, ids, status

    def real_time(self, draw_only_markers=False):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        while True:
            t = time.time()
            _, capture = cap.read()
            if capture is not None:
                original = self.undistort(capture)
<<<<<<< HEAD
                # original = capture
=======
>>>>>>> varm/master
                detected, corners, ids, status = self.detect_markers(original)
                if status:
                    for marker_corner, marker_id in zip(corners, ids):
                        logo = self.objects.get(marker_id[0])
                        original = self.registration.register(original, logo, marker_corner, marker_id)
                    cv2.imshow("Result", original)
                    self.registration.plot_markers(detected, corners, ids) if draw_only_markers else None
                else:
                    pass
                    cv2.imshow("Result", original)
            print("Frame Rate: {} fps".format(round(1 / (time.time() - t), 2)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
