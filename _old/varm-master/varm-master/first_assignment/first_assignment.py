import cv2
import time
import numpy as np
import functools, operator
from first_assignment.first_database.first_database import FirstDatabase
from first_assignment.detection.detection import Detection
from first_assignment.processing.processing import Processing
from first_assignment.recognition.recognition import Recognition


class FirstAssignment:

    def __init__(self, processing: Processing, detection: Detection, r_class: Recognition.__class__, grayscale: bool):
        self.processing = processing
        self.detection = detection
        self.r_class = r_class
        self.grayscale = grayscale
        self.dataset = None

    def setup_resources(self):
        resources = []
        for resource, title in FirstDatabase.load('resources'):
            resources.append(resource)
        self.hat = resources[1].astype(np.uint8)
        self.glasses = resources[0].astype(np.uint8)

    def process_data(self):
        detections = []
        for image, title in FirstDatabase.load('originals'):
            result = self.detection.detect(image)
            if result is not None:
                face, eyes = result
                normalized = self.processing.normalize(eyes, face)
                if self.grayscale:
                    normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
                detections.append((normalized, title))
            else:
                print("No face/eye detection on image file {}".format(title))
        for image, title in detections:
            FirstDatabase.store_normalized(image, title)
        print("Total number of images stored: {}".format(len(detections)))

    def assemble_data(self):
        images = []
        for image, title in FirstDatabase.load('normalized'):
            if self.grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
        shape = functools.reduce(operator.mul, images[0].shape)
        number = len(images)
        self.dataset = np.zeros((shape, number), dtype=np.float32)
        for i in range(number):
            image = images[i].flatten()
            self.dataset[:, i] = image
        print("Dataset dimensions: {}".format(self.dataset.shape))

    def setup_examples(self, original_size: tuple, labels: list, n_examples: int, **kwargs):
        self.recognition = self.r_class(self.dataset, original_size, n_examples, labels, **kwargs)
        for image, title in FirstDatabase.load('examples'):
            face, eyes, image, name = self.recognize(image)
            if image is not None:
                cv2.imshow("Result", image)
                print("Recognized person is " + name)
                print("Press any key next example...")
                cv2.waitKey(0)
        cv2.destroyAllWindows()

    def setup_camera(self, original_size: tuple, labels: list, n_examples: int, **kwargs):
        self.recognition = self.r_class(self.dataset, original_size, n_examples, labels, **kwargs)
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        while True:
            t = time.time()
            _, image = cap.read()
            if image is not None:
                face, eyes, normalized, name = self.recognize(image)
                if face is not None:
                    transformed = self.transform(face, eyes)
                    cv2.putText(transformed, "Face recognition: " + name, (10, transformed.shape[0] - 10), 0, 0.35, (255, 255, 255), 1)
                    cv2.imshow("Result", transformed)
            print("Frame Rate: {} fps".format(round(1 / (time.time() - t), 2)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    def recognize(self, image: np.ndarray, title: str = None):
        result = self.detection.detect(image)
        if result is not None:
            face, eyes = result
            normalized = self.processing.normalize(eyes, face)
            if self.grayscale:
                normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            name = self.recognition.classify(normalized.flatten())
            return face, eyes, normalized, name
        else:
            if title is None:
                print("No face/eye detection on frame capture.")
            else:
                print("No face/eye detection on test image {}.".format(title))
            return None, None, None, "unknown"

    def transform(self, face: np.ndarray, eyes: list):
        try:
            glasses_scale = 50/100
            eyes_offset = (35, 25)
            left_corner = (max(0, eyes[0][0] - eyes_offset[0]), max(0, eyes[0][1] - eyes_offset[1]))
            width = int(face.shape[1] * glasses_scale)
            height = int(self.glasses.shape[0] * width / self.glasses.shape[1])
            glasses_adjusted = cv2.resize(self.glasses, (width, height), interpolation=cv2.INTER_AREA)
            rows, cols, channels = glasses_adjusted.shape
            (x0, y0) = left_corner[0], left_corner[1]
            roi = face[y0: y0 + rows, x0: x0 + cols]
            gray_form = cv2.cvtColor(glasses_adjusted, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_form, 10, 255, cv2.THRESH_BINARY)
            inv_mask = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
            fg = cv2.bitwise_and(glasses_adjusted, glasses_adjusted, mask=mask)
            face[y0: y0 + rows, x0: x0 + cols] = cv2.add(bg, fg)
        except Exception as e:
            print('transform', e)
        return face
