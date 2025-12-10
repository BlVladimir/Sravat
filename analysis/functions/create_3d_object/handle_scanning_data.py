import math
from typing import Tuple, Any

from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions
import numpy as np
from scanning_optimized import scanning_optimized


class HandleScanningData(Function):
    """Из контуров создает массив из центров кубов, которые вместе образуют объект"""
    def __init__(self, state:State, edge:int) -> None:
        super().__init__(state)
        self._EDGE = edge  # Максимальное количество вокселей по одной из осей
        self._THRESHOLD = Config.PHOTO_COUNTS  # Сколько контуров должно указать на отсутсвие куба в точке, чтобы его не учитывать

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        contours = list(map(self._transform_to_local_coordinates, self._state.scanning_data))

        self._logger.info(f'contours length: {len(contours)}')

        main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt, _ = self._state.scanning_data[0]

        parallelepiped = self._calculate_parallelepiped(contours)
        points = scanning_optimized.process_contours_optimized(parallelepiped, contours)

        mask = points < self._THRESHOLD
        self._state.object3d = parallelepiped[mask]

        self._logger.info(f'object3d shape: {self._state.object3d.shape}')

        self._state.scanning_data = []

    @staticmethod
    def _transform_to_local_coordinates(data:Tuple[np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]):
        """
        Преобразует точки в систему координат от диагонали.
        """
        main_vector, auxiliary_vector, origin_point, _ , points_array = data
        main_vec = np.array(main_vector, dtype=np.float32)
        aux_vec = np.array(auxiliary_vector, dtype=np.float32)
        origin = np.array(origin_point, dtype=np.float32)
        points = np.array(points_array, dtype=np.float32)

        if points.ndim == 1:
            points = points.reshape(1, -1)

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

        result_points = np.ascontiguousarray(transformed_points, dtype=np.float32)
        result_R = np.ascontiguousarray(R, dtype=np.float32)


        return result_points, result_R

    def _calculate_parallelepiped(self, contours, margin=1.1):
        """
        Вычисляет параллелепипед на основе всех точек контуров
        """
        all_points = []
        for contour_points, _ in contours:
            all_points.extend(contour_points)

        all_points = np.array(all_points, dtype=np.float32)

        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)

        center = (min_coords + max_coords) / 2
        sizes = max_coords - min_coords

        sizes_with_margin = sizes * margin

        min_coords_margin = center - sizes_with_margin / 2
        max_coords_margin = center + sizes_with_margin / 2

        max_size = np.max(sizes_with_margin)
        step = max_size / self._EDGE
        self._state.cube_side = step

        num_cells_x = self._special_round((max_coords_margin[0] - min_coords_margin[0]) / step)
        num_cells_y = self._special_round((max_coords_margin[1] - min_coords_margin[1]) / step)
        num_cells_z = self._special_round((max_coords_margin[2] - min_coords_margin[2]) / step)

        self._logger.info(
            f"Grid size: {num_cells_x} x {num_cells_y} x {num_cells_z} = {num_cells_x * num_cells_y * num_cells_z} cells")

        parallelepiped = np.array([
            [
                min_coords_margin[0] + (x + 0.5) * step,
                min_coords_margin[1] + (y + 0.5) * step,
                min_coords_margin[2] + (z + 0.5) * step
            ]
            for x in range(num_cells_x)
            for y in range(num_cells_y)
            for z in range(num_cells_z)
        ], dtype=np.float32)

        return parallelepiped

    @staticmethod
    def _special_round(x, direction:str='ceil'):
        """Округление с порогом, чтобы float погрешность не добавляла точек"""
        threshold = 1e-6
        if direction == 'ceil':
            return math.ceil(x - threshold)
        else:
            return math.floor(x + threshold)