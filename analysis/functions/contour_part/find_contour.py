from functools import partial

from analysis.analysis_state import Method
from analysis.functions.function import Function, handle_exceptions

import numpy as np
import cv2


class FindContour(Function):
    """Обрабатывает кадр и рисует контуры только внутри плоскости"""
    def __init__(self, state):
        super().__init__(state)
        self.area_threshold = 500

    @staticmethod
    def is_contour_inside_plane(plane_points, contour):
        """Проверяет, полностью ли контур находится внутри плоскости"""
        plane_contour = np.array(plane_points, dtype=np.float32).reshape(-1, 1, 2)

        contour_points = contour.reshape(-1, 2)

        border_threshold = 1.0

        for point in contour_points:
            result = cv2.pointPolygonTest(plane_contour, tuple(point.astype(float)), False)
            if result < border_threshold:
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

        src_points = self._state.src_points

        filter_contours = filter(partial(self.is_contour_inside_plane, src_points), contours)

        if filter_contours:
            self._state.contour = max(filter_contours, key=cv2.contourArea)
        else:
            self._state.method = Method.ERROR