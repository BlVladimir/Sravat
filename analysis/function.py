from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

import cv2
import numpy as np

from analysis.analysis_environment import Environment


class Function(ABC):
    """Функции, на которые разбивается алгоритм"""
    def __init__(self, environment:Environment):
        self.logger = getLogger(type(self).__name__)
        self.environment = environment

    @abstractmethod
    def __call__(self, *args, **kwargs)->Any:
        pass

class FindRect(Function):
    """Алгоритм поиска прямоугольника"""
    def __init__(self, environment:Environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        frame = self.environment.current_frame
        try:
            corners, ids, rejected = self.environment.detector.detectMarkers(frame)
            if ids is not None and len(ids) == 4:
                all_corners = np.vstack([c[0] for c in corners])

                x, y, w, h = cv2.boundingRect(all_corners)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            self.environment.current_frame = frame
        except Exception as e:
            self.logger.exception(f'Error finding rect on frame: {e}')