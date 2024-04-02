import os
import cv2
import numpy as np

from utils.utils import Utils

class Filter:

    def __init__(self):
        self.__filters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\database\\filters\\")

    def apply_filter(self, orginal_image, id, eyes):

        self.__load_filter(id)

        eyes_distance = np.sqrt((eyes[0]['x'] - eyes[1]['x'])**2 + (eyes[0]['y'] - eyes[1]['y'])**2)

        eyes_angle = np.degrees(np.arctan2(-(eyes[1]['y'] - eyes[0]['y']), eyes[1]['x'] - eyes[0]['x']))

        rotation_matrix = cv2.getRotationMatrix2D((self.__filter_eyes[0]['x'], self.__filter_eyes[0]['y']), eyes_angle, 1.0)

        filter_rotated = cv2.warpAffine(self.__filter, rotation_matrix, (self.__filter.shape[1], self.__filter.shape[0]))

        eyes_distance_filter = np.sqrt((self.__filter_eyes[0]['x'] - self.__filter_eyes[1]['x'])**2 + (self.__filter_eyes[0]['y'] - self.__filter_eyes[1]['y'])**2)

        scale_factor = eyes_distance / eyes_distance_filter

        filter_resized = cv2.resize(filter_rotated, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_AREA)

        self.__filter_eyes[0]['x'] = int(self.__filter_eyes[0]['x']*scale_factor)
        self.__filter_eyes[0]['y'] = int(self.__filter_eyes[0]['y']*scale_factor)
        self.__filter_eyes[1]['x'] = int(self.__filter_eyes[1]['x']*scale_factor)
        self.__filter_eyes[1]['y'] = int(self.__filter_eyes[1]['y']*scale_factor)
        
        filter_gray = cv2.cvtColor(filter_resized, cv2.COLOR_BGR2GRAY)

        ret, mask = cv2.threshold(filter_gray, 100, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        roi = orginal_image[eyes[0]['y'] - self.__filter_eyes[0]['y']: eyes[0]['y'] - self.__filter_eyes[0]['y'] + filter_resized.shape[0], 
                eyes[0]['x'] - self.__filter_eyes[0]['x']: eyes[0]['x'] - self.__filter_eyes[0]['x'] + filter_resized.shape[1]]

        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(filter_resized, filter_resized, mask=mask)
        
        #roi = cv2.add(bg, fg)
        orginal_image[eyes[0]['y'] - self.__filter_eyes[0]['y']: eyes[0]['y'] - self.__filter_eyes[0]['y'] + filter_resized.shape[0], 
                eyes[0]['x'] - self.__filter_eyes[0]['x']: eyes[0]['x'] - self.__filter_eyes[0]['x'] + filter_resized.shape[1]] = cv2.add(bg, fg)

        return orginal_image


    def __load_filter(self, id):
        if id == 'anaOliveira':
            self.__filter = Utils.read_image(self.__filters_path + 'nervous_eyes.png', style='rgb') 
            self.__filter = cv2.cvtColor(self.__filter, cv2.COLOR_RGB2BGR)

            self.__filter_eyes = ({'x': 95, 'y': 115}, 
                                  {'x': 200, 'y': 115}) # (left_eye, right_eye)
        elif id == 'pedroMjorge':
            self.__filter = Utils.read_image(self.__filters_path + 'green_eyes.png', style='rgb') 
            self.__filter = cv2.cvtColor(self.__filter, cv2.COLOR_RGB2BGR)

            self.__filter_eyes = ({'x': 95, 'y': 100}, 
                                  {'x': 195, 'y': 100}) # (left_eye, right_eye)
        else:
            self.__filter = Utils.read_image(self.__filters_path + 'dragon_eyes.png', style='rgb') 
            self.__filter = cv2.cvtColor(self.__filter, cv2.COLOR_RGB2BGR)
            self.__filter_eyes = ({'x': 77, 'y': 84}, 
                                  {'x': 210, 'y': 84}) # (left_eye, right_eye)
