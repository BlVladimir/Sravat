from analysis.analysis_config import Config
from analysis.functions.function import Function, handle_exceptions

import cv2
import numpy as np


class Adaptive(Function):
    def __init__(self, state):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self._state.current_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        pts = np.int32(self._state.src_points)
        cv2.fillPoly(mask, [pts], 255)

        # Применяем адаптивный порог
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # размер блока для вычисления порога
            2  # константа вычитания из среднего
        )

        binary = cv2.bitwise_and(binary, binary, mask=mask)

        # Морфологические операции для очистки от шума
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # убираем шум
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # заполняем дыры

        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        min_area = 100
        approximation_epsilon = 0.008

        # Фильтруем контуры по площади
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            return

        main_contour = max(valid_contours, key=cv2.contourArea)

        # Аппроксимируем контур (упрощаем)
        epsilon = approximation_epsilon * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)

        if approx_contour is not None:
            cv2.drawContours(frame, [approx_contour], -1, Config.colors['contour'], 3)
            self._state.current_frame =  frame