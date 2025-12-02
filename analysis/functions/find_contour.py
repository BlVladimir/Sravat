from analysis.functions.function import Function, handle_exceptions

import numpy as np
import cv2


class FindContour(Function):
    """Обрабатывает кадр и рисует контуры только внутри плоскости"""
    def __init__(self, state):
        super().__init__(state)
        self.area_threshold = 500

    @staticmethod
    def is_contour_inside_plane(contour, plane_points):
        """Проверяет, полностью ли контур находится внутри плоскости"""
        if plane_points is None or len(plane_points) != 4:
            return False

        # Преобразуем plane_points в правильный формат для OpenCV
        plane_contour = np.array(plane_points, dtype=np.float32).reshape(-1, 1, 2)

        # Преобразуем контур в массив точек
        contour_points = contour.reshape(-1, 2)

        # Проверяем каждую точку контура
        for point in contour_points:
            # Проверяем, находится ли точка внутри четырехугольника
            result = cv2.pointPolygonTest(plane_contour, tuple(point.astype(float)), False)
            if result < 0:
                return False
        return True

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self._state.current_frame
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh = cv2.threshold(img_blur, 155, 200, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)

        self._state.current_frame = frame
        self._state.contour = contour