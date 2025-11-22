from analysis.analysis_config import Config
from analysis.analysis_environment import Environment
from analysis.functions.function import Function

import numpy as np
import cv2


class DetectMarkers(Function):
    """Детектирует ArUco маркеры и возвращает их центры и углы"""

    def __init__(self, environment: Environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        frame = self.environment.current_frame
        corners, ids, rejected = self.environment.detector.detectMarkers(frame)

        if ids is None:
            return frame, [], []

        # Рисуем обнаруженные маркеры
        output_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)

        # Собираем центры маркеров
        centers = []
        for corner in corners:
            center = np.mean(corner[0], axis=0)
            centers.append(center)
            # Рисуем центр маркера
            cv2.circle(output_frame, tuple(center.astype(int)), 3, Config.colors['center'], -1)

        self.environment.current_frame = output_frame
        self.environment.centers = centers

        return corners