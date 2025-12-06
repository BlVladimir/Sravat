import math
from typing import Tuple, Any

from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function
import numpy as np
import scanning_optimized


class HandleScanningData(Function):
    def __init__(self, state:State) -> None:
        super().__init__(state)

    def __call__(self, *args, **kwargs):
        contours = map(self._transform_to_local_coordinates, self._state.contour)
        main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt, _ = self._state.scanning_data[0]
        parallelepiped = self._calculate_parallelepiped(main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt)
        self._state.scanning_data = []

    @staticmethod
    def _transform_to_local_coordinates(data:Tuple[np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]):
        """Преобразует точки в систему координат от диагонали."""
        main_vector, auxiliary_vector, origin_point, _ , points_array = data
        main_vec = np.array(main_vector, dtype=float)
        aux_vec = np.array(auxiliary_vector, dtype=float)
        origin = np.array(origin_point, dtype=float)
        points = np.array(points_array, dtype=float)

        scale = np.linalg.norm(main_vec)

        x_axis = main_vec / scale

        z_axis = np.cross(main_vec, aux_vec)
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        rotation_matrix = np.array([
            x_axis,
            y_axis,
            z_axis
        ])

        shifted_points = points - origin

        transformed_points = shifted_points @ rotation_matrix.T
        transformed_points = transformed_points / scale

        normal = np.array([0, 0, -1], dtype=np.float32) @ rotation_matrix.T
        normal /= np.linalg.norm(normal)
        target = np.array([0, 0, 1])

        axis = np.cross(normal, target)
        axis_norm = np.linalg.norm(axis)

        axis = axis / axis_norm

        cos_angle = np.dot(normal, target)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)

        I = np.eye(3, dtype=np.float32)
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        return transformed_points, R

    @staticmethod
    def _calculate_parallelepiped(main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt):
        """Создание параллелепипеда, из которого будет вырезан объект"""
        norm1 = float(np.linalg.norm(main_vec))
        norm2 = float(np.linalg.norm(auxiliary_vec))
        cos_alpha = (main_vec @ auxiliary_vec)/(norm1 * norm2)
        sin_alpha = np.sqrt(1 - cos_alpha**2)
        b = (norm2 / norm1)*sin_alpha
        h = min(1, b)

        r = origin_auxiliary_pnt - origin_main_pnt
        norm1 = float(np.linalg.norm(main_vec))
        norm2 = float(np.linalg.norm(r))
        cos_alpha = (main_vec @ r) / (norm1 * norm2)
        sin_alpha = np.sqrt(1 - cos_alpha ** 2)

        y0 = -(norm2/norm1)*sin_alpha

        step = h/Config.EDGE

        parallelepiped = np.array([[(x+0.5)*step, (y+0.5)*step, (z+0.5)*step]
                                    for x in range(0, math.ceil(1/step))
                                    for y in range(math.floor(y0 / step), math.ceil((y0 + b) / step))
                                    for z in range(0, math.ceil(h / step)+1)])

        return parallelepiped

