import numpy as np

from first_assignment.detection.caffe_detection import CaffeDetection
from first_assignment.detection.haar_detection import HaarDetection


class Detection:

    def __init__(self, face_detection: CaffeDetection, eye_detection: HaarDetection) -> None:
        self.face_detection = face_detection
        self.eye_detection = eye_detection

    def detect(self, image: np.ndarray) -> np.ndarray:
        face_image, roi, box = self.face_detection.face_detect(image)
        if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
            return self.eye_detection.eye_detect(roi)