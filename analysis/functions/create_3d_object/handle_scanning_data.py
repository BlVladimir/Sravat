import math
from typing import Tuple, Any

from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions
import numpy as np
from scanning_optimized import scanning_optimized


class HandleScanningData(Function):
    def __init__(self, state:State, edge:int) -> None:
        super().__init__(state)
        self._EDGE = edge
        self._THRESHOLD = 3

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        """Из контуров создает массив из центров кубов, которые вместе образуют объект"""
        contours = list(map(self._transform_to_local_coordinates, self._state.scanning_data))

        main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt, _ = self._state.scanning_data[0]
        parallelepiped = self._calculate_parallelepiped(main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt)
        points = scanning_optimized.process_contours_optimized(parallelepiped, contours)

        self._logger.debug(f'Number of contours: {len(contours)}')
        self._logger.debug(f'Parallelepiped shape: {parallelepiped.shape}')
        self._logger.debug(f'Parallelepiped range: X[{parallelepiped[:, 0].min():.3f}, {parallelepiped[:, 0].max():.3f}], '
                          f'Y[{parallelepiped[:, 1].min():.3f}, {parallelepiped[:, 1].max():.3f}], '
                          f'Z[{parallelepiped[:, 2].min():.3f}, {parallelepiped[:, 2].max():.3f}]')

        first_contour_points, first_rotation = contours[0]
        self._logger.debug(f'First contour points shape: {first_contour_points.shape}')
        self._logger.debug(f'First contour range BEFORE rotation: '
                           f'X[{first_contour_points[:, 0].min():.3f}, {first_contour_points[:, 0].max():.3f}], '
                           f'Y[{first_contour_points[:, 1].min():.3f}, {first_contour_points[:, 1].max():.3f}], '
                           f'Z[{first_contour_points[:, 2].min():.3f}, {first_contour_points[:, 2].max():.3f}]')

        rotated = first_contour_points @ first_rotation.T
        self._logger.debug(f'First contour range AFTER rotation: '
                           f'X[{rotated[:, 0].min():.3f}, {rotated[:, 0].max():.3f}], '
                           f'Y[{rotated[:, 1].min():.3f}, {rotated[:, 1].max():.3f}], '
                           f'Z[{rotated[:, 2].min():.3f}, {rotated[:, 2].max():.3f}]')

        mask = points <= self._THRESHOLD
        self._logger.debug(f'Mask: {mask}')
        self._logger.debug(f'Points: {points}')

        self._state.object3d = parallelepiped[mask]
        self._state.scanning_data = []

    @staticmethod
    def _transform_to_local_coordinates(data:Tuple[np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]):
        """Преобразует точки в систему координат от диагонали."""
        main_vector, auxiliary_vector, origin_point, _ , points_array = data
        main_vec = np.array(main_vector, dtype=np.float32)
        aux_vec = np.array(auxiliary_vector, dtype=np.float32)
        origin = np.array(origin_point, dtype=np.float32)
        points = np.array(points_array, dtype=np.float32)

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

        if axis_norm < 1e-6:
            cos_angle = np.dot(normal, target)
            if cos_angle > 0:
                R = np.eye(3, dtype=np.float32)
            else:
                if abs(normal[0]) < 0.9:
                    axis = np.array([1, 0, 0], dtype=np.float32)
                else:
                    axis = np.array([0, 1, 0], dtype=np.float32)
                axis = np.cross(normal, axis)
                axis /= np.linalg.norm(axis)

                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]], dtype=np.float32)
                I = np.eye(3, dtype=np.float32)
                R = I + 2 * (K @ K)
        else:
            axis = axis / axis_norm

            cos_angle = np.dot(normal, target)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=np.float32)

            I = np.eye(3, dtype=np.float32)
            R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        return np.ascontiguousarray(transformed_points, dtype=np.float32), np.ascontiguousarray(R, dtype=np.float32)

    def _calculate_parallelepiped(self, main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt):
        """Создание параллелепипеда, из которого будет вырезан объект"""
        norm1 = float(np.linalg.norm(main_vec))
        norm2 = float(np.linalg.norm(auxiliary_vec))
        cos_alpha = (main_vec @ auxiliary_vec)/(norm1 * norm2)
        sin_alpha = np.sqrt(1 - cos_alpha**2)
        b = (norm2 / norm1)*sin_alpha
        h = min(1, b)

        r = origin_auxiliary_pnt - origin_main_pnt
        norm2_r = float(np.linalg.norm(r))
        cos_alpha_r = (main_vec @ r) / (norm1 * norm2_r)
        sin_alpha_r = np.sqrt(1 - cos_alpha_r ** 2)

        y0 = -(norm2_r/norm1)*sin_alpha_r

        step = max(1, b)/self._EDGE
        self._state.cube_side = step

        parallelepiped = np.array([[(x + 0.5) * step, (y + 0.5) * step, (z + 0.5) * step]
                                   for x in range(0, self._special_round(1 / step))
                                   for y in range(self._special_round(y0 / step, 'floor'), self._special_round((y0 + b) / step))
                                   for z in range(0, self._special_round(h / step))], dtype=np.float32)

        return parallelepiped

    @staticmethod
    def _special_round(x, direction:str='ceil'):
        """Округление с порогом, чтобы float погрешность не добавляла точек"""
        threshold = 1e-6
        if direction == 'ceil':
            return math.ceil(x - threshold)
        else:
            return math.floor(x + threshold)