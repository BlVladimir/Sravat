from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions
import numpy as np


class CreateHomographyTransform(Function):
    """
    Расчет уравнения плоскости четырехугольника. Название связано с историей создания функции
    """
    def __init__(self, state:State):
        super().__init__(state)
        self._EPSILON = 1e-10

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        points_3d = np.array(
            [np.array(self._state.marker_data[marker_id]['tvec'], dtype=np.float32)
             for marker_id in self._state.marker_data.keys()],
            dtype=np.float32
        )

        normals = []
        valid_normals_count = 0

        for i in range(4):
            p1, p2, p3 = points_3d[i], points_3d[(i + 1) % 4], points_3d[(i + 2) % 4]
            v1 = p2 - p1
            v2 = p3 - p1

            if np.linalg.norm(v1) < self._EPSILON or np.linalg.norm(v2) < self._EPSILON:
                continue

            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < self._EPSILON:
                continue

            normals.append(normal / norm)
            valid_normals_count += 1

        if valid_normals_count == 0:
            self._state.plane_equation = None
            return

        average_normal = np.mean(normals, axis=0)
        norm_average = np.linalg.norm(average_normal)

        if norm_average < self._EPSILON:
            self._state.plane_equation = None
            return

        average_normal /= norm_average

        centroid = np.mean(points_3d, axis=0)

        distance = -np.dot(average_normal, centroid)

        self._state.plane_equation = (average_normal, distance)