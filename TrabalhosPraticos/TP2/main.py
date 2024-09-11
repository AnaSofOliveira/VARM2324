from spikes.camera_calibration import calibrate
from spikes.registration import *

#calibrate()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if ret:
        frame_markers, corners, ids, rejectedImgPoints = detect_markers(frame, show_axis=True)

        if len(corners) > 0:
            for corner, marker_id in zip(corners, ids):               
                frame_markers = register(frame, corner, marker_id)

            #plot_markers(frame_markers, corners, ids)
        cv2.imshow('Frame', frame_markers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()  