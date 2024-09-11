import cv2
from typing import Final

# Chessboard configurations
CHESSBOARD_ROWS: Final = 7
CHESSBOARD_COLS: Final = 6
CHESSBOARD_SQUARE_SIZE: Final = 1

# Aruco Markers configurations
MAX_MARKER_ID: Final = 10
ARUCO_MARKER_SIZE: Final = 50
ARUCO_MARKER_DICT: Final = cv2.aruco.DICT_4X4_50