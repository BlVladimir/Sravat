from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions

import cv2
import numpy as np


class DrawPlane(Function):
    """Рисует плоскость и маркеры на кадре"""
    def __init__(self, state: State):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self._state.current_frame
        src_points = self._state.src_points

        pts = src_points.astype(int)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        cv2.polylines(frame, [pts], True, Config.COLORS['contour'], 3)

        cv2.addWeighted(frame, 0.2, frame, 0.8, 0, frame)

        for i, (point, color) in enumerate(zip(pts, Config.COLORS['corners'])):
            cv2.circle(frame, tuple(point), 3, color, -1)

        self._state.current_frame = frame