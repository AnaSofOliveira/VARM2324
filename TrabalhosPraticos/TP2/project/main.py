import cv2
from register.Register import Register
from calibration.Calibration import Calibration

calibration = Calibration()
register = Register(calibration)


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if ret:
        frame_markers, corners, ids, rejectedImgPoints = register.detect_aruco_markers(frame, show_axis=True)

        if len(corners) > 0:
            for markers, marker_id in zip(corners, ids):               
                frame_markers = register.register_images_on_markers(frame, markers, marker_id)

        cv2.imshow('Frame', frame_markers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()  
