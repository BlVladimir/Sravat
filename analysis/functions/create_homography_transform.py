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
        centers = self._state.centers

        # Сортируем точки
        src_points = np.float32(self.sort_points(centers))

        self._state.src_points = src_points

        points_3d = np.array(
            [np.array(self._state.marker_data[marker_id]['tvec'], dtype=np.float32)
             for marker_id in self._state.marker_data.keys()],
            dtype=np.float32
        )

        normals = []
        for i in range(4):
            # Выбираем три точки
            p1, p2, p3 = points_3d[i], points_3d[(i + 1) % 4], points_3d[(i + 2) % 4]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normals.append(normal / np.linalg.norm(normal))

        average_normal = np.mean(normals, axis=0)
        average_normal /= np.linalg.norm(average_normal)

        centroid = np.mean(points_3d, axis=0)

        # Расстояние до плоскости
        distance = -np.dot(average_normal, centroid)

        self._state.plane_equation = (average_normal, distance)
