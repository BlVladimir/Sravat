from analysis.analysis_environment import State
from analysis.functions.function import Function, handle_exceptions

import numpy as np

class CreateHomographyTransform(Function):
    """Создает гомографию для преобразования плоскости"""

    @staticmethod
    def sort_points(points):
        """Сортирует точки в порядке: top-left, top-right, bottom-right, bottom-left"""
        points = np.array(points)

        # Сортируем по y-координате
        y_sorted = points[np.argsort(points[:, 1])]

        # Верхние две точки
        top_points = y_sorted[:2]
        # Нижние две точки
        bottom_points = y_sorted[2:]

        # Сортируем верхние точки по x
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        tl, tr = top_sorted[0], top_sorted[1]

        # Сортируем нижние точки по x
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        bl, br = bottom_sorted[0], bottom_sorted[1]

        return [tl, tr, br, bl]

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        """Создает гомографию для преобразования плоскости"""
        centers = self.env.centers

        if len(centers) != 4:
            self.env.centers = []
            self.env.src_points = None
            self.env.state = State.END
            return
        # Сортируем точки
        src_points = np.float32(self.sort_points(centers))

        self.env.src_points = src_points