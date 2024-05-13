import cv2
import math
import itertools
import numpy as np


class Processing:

    LEFT_EYE = (16, 24)
    RIGHT_EYE = (31, 24)
    HORIZONTAL_DIFFERENCE = 15
    SIZE = (56, 46)

    def __init__(self, size: tuple = SIZE):
        self.size = size
        h_ratio = size[1] / self.SIZE[1]
        v_ratio = size[0] / self.SIZE[0]
        self.desired_left_eye = self.LEFT_EYE[0] * h_ratio, self.LEFT_EYE[1] * v_ratio
        self.desired_right_eye = self.RIGHT_EYE[0] * h_ratio, self.RIGHT_EYE[1] * v_ratio
        self.desired_difference = self.HORIZONTAL_DIFFERENCE * h_ratio

    def normalize(self, correct_eyes, image):
        angle = correct_eyes[2][0]
        detected_left_eye, detected_right_eye = correct_eyes[0], correct_eyes[1]
        horizontal_difference = detected_right_eye[0] - detected_left_eye[0]
        eyes_scale = self.desired_difference / horizontal_difference
        horizontal_translation = self.desired_left_eye[0] - detected_left_eye[0]
        vertical_translation = self.desired_left_eye[1] - detected_left_eye[1]
        translation_matrix = np.float32([[1, 0, horizontal_translation], [0, 1, vertical_translation], [0, 0, 1]])
        rotation_matrix = np.vstack((cv2.getRotationMatrix2D(self.desired_left_eye, np.float32(angle), eyes_scale), np.float32([[0, 0, 1]])))
        matrix = np.dot(rotation_matrix, translation_matrix)[:-1, :]
        return cv2.warpAffine(image, matrix, self.size[::-1])

    @staticmethod
    def validate_eyes(eye_points, max_angle):
        min_distance = np.inf
        best_pair = None
        for pair in list(itertools.combinations(eye_points, 2)):
            left_eye, right_eye = (pair[0], pair[1]) if pair[0][0] < pair[1][0] else (pair[1], pair[0])
            horizontal_overlapping = left_eye[0] < right_eye[0] < left_eye[0] + (left_eye[2] / 2)
            if not horizontal_overlapping:
                horizontal_distance = abs(left_eye[0] - right_eye[0])
                vertical_distance = abs(left_eye[1] - right_eye[1])
                overall_distance = vertical_distance - horizontal_distance
                if overall_distance < min_distance:
                    min_distance = overall_distance
                    best_pair = [left_eye, right_eye, None]
        if best_pair is not None:
            angle = Processing.get_angle(best_pair[0], best_pair[1])
            if abs(angle) < max_angle:
                best_pair[2] = (angle,)
                return best_pair

    @staticmethod
    def draw_eyes(eyes, image):
        for i in range(2):
            (center_x, center_y, radius_w, radius_h) = eyes[i]
            center_point = (center_x, center_y)
            radius = radius_w // 10
            image = cv2.circle(image, center_point, radius, (0, 255, 0), 1)

    @staticmethod
    def get_angle(p1, p2):
        r = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return min(math.degrees(r), 180 - math.degrees(r))