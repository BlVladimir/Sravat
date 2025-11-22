from analysis.analysis_config import Config
from analysis.analysis_environment import Environment, State
from analysis.functions.function import Function

import numpy as np
import cv2


class DetectMarkers(Function):
    """Детектирует ArUco маркеры и возвращает их центры и углы"""

    def __init__(self, environment: Environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        try:
            frame = self.env.current_frame
            corners, ids, rejected = self.env.detector.detectMarkers(frame)

            if ids is None:
                return

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

            self.env.current_frame = output_frame
            self.env.centers = centers
        except Exception as e:
            self.logger.error(f'Error detecting markers: {e}')
            self.env.state = State.ERROR