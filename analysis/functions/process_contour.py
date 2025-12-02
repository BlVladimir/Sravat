from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions

import numpy as np


class ProcessContour(Function):
    """Переводит контур в 3D координаты"""
    def __init__(self, state:State):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        self._state.current_contour_3d.clear()

        if self._state.plane_equation is not None and Config.camera_matrix is not None:
            contour_3d = []
            for point in self._state.contour.reshape(-1, 2):
                point_3d = self.project_2d_to_3d(
                    tuple(point),
                    Config.camera_matrix,
                    self._state.plane_equation
                )
                if point_3d is not None:
                    contour_3d.append(point_3d)
            self._state.current_contour_3d.append(contour_3d)

    @staticmethod
    def project_2d_to_3d(point_2d, camera_matrix, plane_equation):
        """Проецирует 2D точку изображения в 3D на заданной плоскости"""
        n, d = plane_equation
        # Обратная проекция: из 2D в луч в 3D
        inv_K = np.linalg.inv(camera_matrix)
        point_2d_hom = np.array([point_2d[0], point_2d[1], 1.0])
        ray_dir = inv_K.dot(point_2d_hom)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)  # нормализуем

        # Находим пересечение луча с плоскостью: t = - (n·O + d) / (n·ray_dir)
        # Точка O (начало луча) - это центр камеры (0,0,0) в системе координат камеры
        t = -d / np.dot(n, ray_dir)
        point_3d = t * ray_dir
        return point_3d
