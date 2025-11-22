from analysis.functions.function import Function

import cv2
import numpy as np


class CannyMethod(Function):
    def __init__(self, environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        min_area = 500
        approximation_epsilon = 0.008
        frame = self.env.current_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Вычисляем резкость изображения через лапласиан
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        sigma = 0.33
        median_val = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))

        edges = cv2.Canny(gray, lower, upper)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Находим контуры
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        # Фильтруем контуры по площади
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            return

        # Берем самый большой контур
        main_contour = max(valid_contours, key=cv2.contourArea)

        # Аппроксимируем контур (упрощаем)
        epsilon = approximation_epsilon * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)

        result_frame = frame.copy()

        if approx_contour is not None:
            # Рисуем КРАСНЫЙ контур поверх исходной картинки
            cv2.drawContours(result_frame, [approx_contour], -1, (0, 0, 255), 3)
            self.env.current_frame =  result_frame