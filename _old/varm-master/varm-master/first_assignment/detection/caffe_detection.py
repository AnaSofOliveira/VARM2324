import cv2
import numpy as np


class CaffeDetection:

    MIN_CONFIDENCE = 0.75

    def __init__(self):
        super().__init__()
        self.model = "./first_assignment/models/res10_300x300_ssd_iter_140000.caffemodel"
        self.config = "./first_assignment/models/deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(self.config, self.model)

    def face_detect(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        roi_color = None
        (x0, y0, x1, y1) = (None, None, None, None)
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > CaffeDetection.MIN_CONFIDENCE:
                box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x0, y0, x1, y1) = box.astype(int)
                box_width = (x1 - x0)
                box_height = (y1 - y0)
                margin_x0 = max(0, x0 - box_width//2)
                margin_x1 = min(width, x1 + box_width//2)
                margin_y0 = max(0, y0 - box_height//2)
                margin_y1 = min(height, y1 + box_height//2)
                roi_color = image[margin_y0: margin_y1, margin_x0: margin_x1]
                # cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
        return image, roi_color, (x0, y0, x1, y1)
