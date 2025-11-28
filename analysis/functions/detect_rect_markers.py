from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions

import numpy as np
import cv2


class DetectRectMarkers(Function):
    """Детектирует ArUco маркеры и возвращает их центры и углы"""

    def __init__(self, state: State):
        super().__init__(state)

        self.aruco_rect_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_rect_params = cv2.aruco.DetectorParameters()
        self.detector_rect_markers = cv2.aruco.ArucoDetector(self.aruco_rect_dict, self.aruco_rect_params)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self.state.current_frame
        corners, ids, rejected = self.detector_rect_markers.detectMarkers(frame)

        if ids is None:
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Собираем центры маркеров
        centers = []
        for corner in corners:
            center = np.mean(corner[0], axis=0)
            centers.append(center)
            # Рисуем центр маркера
            cv2.circle(frame, tuple(center.astype(int)), 3, Config.colors['center'], -1)

        self.state.current_frame = frame
        self.state.centers = centers