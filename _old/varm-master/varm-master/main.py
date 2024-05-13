import argparse
import configparser

from first_assignment.detection.detection import Detection
from first_assignment.first_assignment import FirstAssignment
from first_assignment.processing.processing import Processing
from first_assignment.recognition.eigen_faces import EigenFaces
from first_assignment.recognition.fisher_faces import FisherFaces
from first_assignment.detection.haar_detection import HaarDetection
from first_assignment.detection.caffe_detection import CaffeDetection
from second_assignment.calibration.calibration import Calibration
from second_assignment.registration.registration import Registration
from second_assignment.second_assignment import SecondAssignment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="Run configuration. Please choose configuration file path.")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.f)

    p = config['Project']['type']
    t = config['FacialRecognition']['type']
    o = config['Operation']['mode']
    m = config['Markers']['mode']

    if p == 'feature-based':
        draw_only_markers = False
        gray = True
        res = (56, 46)
        examples = 10
        variance = 0.95
        size = res if gray else (res[0], res[1], 3)
        labels = ['Angela Merkel', 'Angelina Jolie', 'António Costa', 'Barack Obama', 'Cristiano Ronaldo', 'Eduardo Santos', 'Pedro Costa']
        if t == "fisher":
            t = FisherFaces
        elif t == "eigen":
            t = EigenFaces
        p = Processing(res)
        d = Detection(CaffeDetection(), HaarDetection(draw_only_markers))
        a = FirstAssignment(p, d, t, gray)
        a.process_data()
        a.assemble_data()
        a.setup_resources()
        if o == "real-time":
            a.setup_camera(size, labels, examples, variance=variance)
        elif o == "first_database":
            a.setup_examples(size, labels, examples, variance=variance)
    elif p == 'marker-based':
        image_format = 'jpg'
        square_size = 2.4
        width = 6
        height = 8
        t = Registration()
        c = Calibration(image_format, square_size, width, height)
        a = SecondAssignment(c, t)
        draw_only_markers = True if m == 'yes' else False
        a.real_time(draw_only_markers)
