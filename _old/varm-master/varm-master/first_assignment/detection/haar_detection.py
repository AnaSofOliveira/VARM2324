import cv2
import numpy as np
from first_assignment.processing.processing import Processing


class HaarDetection:

    MAX_ANGLE = 45

    def __init__(self, draw: False = bool):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.draw = draw

    def face_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        roi_color = image
        for (x, y, w, h) in faces:
            # image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = image[y: y + h, x: x + w]
        return image, roi_color

    def eye_detect(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = self.eyes_cascade.detectMultiScale(gray)
        if type(eyes) == np.ndarray and len(eyes) > 1:
            eye_points = [(ex + ew // 2, ey + eh // 2, ew, eh) for (ex, ey, ew, eh) in eyes]
            correct_eyes = Processing.validate_eyes(eye_points, HaarDetection.MAX_ANGLE)
            if correct_eyes is not None and len(correct_eyes) >= 2:
                if self.draw:
                    Processing.draw_eyes(correct_eyes, image)
                return image, correct_eyes
