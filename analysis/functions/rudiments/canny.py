from analysis.analysis_config import Config
from analysis.functions.function import Function, handle_exceptions

import cv2
import numpy as np


class CannyMethod(Function):
    def __init__(self, state):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self._state.current_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        pts = np.int32(self._state.src_points)
        cv2.fillPoly(mask, [pts], 255)

        # Применяем маску: оставляем только пиксели внутри четырёхугольника
        gray = cv2.bitwise_and(gray, gray, mask=mask)

        masked_pixels = gray[mask == 255]
        median_val = np.median(masked_pixels)

        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))

        edges = cv2.Canny(gray, lower, upper)

        edges = cv2.bitwise_and(edges, edges, mask=mask)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Находим контуры
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        min_area = 100
        approximation_epsilon = 0.008

        # Фильтруем контуры по площади
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            return

        main_contour = max(valid_contours, key=cv2.contourArea)

        # Аппроксимируем контур
        epsilon = approximation_epsilon * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)

        if approx_contour is not None:
            cv2.drawContours(frame, [approx_contour], -1, Config.COLORS['contour'], 3)
            self._state.current_frame =  frame