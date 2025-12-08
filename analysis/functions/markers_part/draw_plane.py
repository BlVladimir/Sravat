from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions

import cv2
import numpy as np


class DrawPlane(Function):
    def __init__(self, state: State):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        """Рисует плоскость и маркеры на кадре"""
        frame = self._state.current_frame
        src_points = self._state.src_points

        pts = src_points.astype(int)

        # Создаём маску четырёхугольника
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Инвертируем маску (всё вне четырёхугольника)
        mask_inverted = cv2.bitwise_not(mask)

        # Заливаем всё вне четырёхугольника белым
        frame[mask_inverted == 255] = [255, 255, 255]

        # Рисуем контур плоскости
        cv2.polylines(frame, [pts], True, Config.COLORS['contour'], 3)

        cv2.addWeighted(frame, 0.2, frame, 0.8, 0, frame)

        # Рисуем угловые точки
        for i, (point, color) in enumerate(zip(pts, Config.COLORS['corners'])):
            cv2.circle(frame, tuple(point), 3, color, -1)

        self._state.current_frame = frame