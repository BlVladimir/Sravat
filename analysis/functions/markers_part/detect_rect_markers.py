from itertools import repeat

from analysis.analysis_config import Config
from analysis.analysis_state import State, Method
from analysis.functions.function import Function, handle_exceptions

import numpy as np
import cv2


class DetectRectMarkers(Function):
    """Детектирует ArUco маркеры и возвращает их центры и углы"""

    def __init__(self, state: State):
        super().__init__(state)

        self.aruco_rect_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_rect_params = cv2.aruco.DetectorParameters()
        self.detector_rect_markers = cv2.aruco.ArucoDetector(self.aruco_rect_dict, self.aruco_rect_params)

        self._ids_diag = []

        self._smoothed_tvecs = {}
        self._smoothing_alpha = 0.3

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self._state.current_frame
        corners, ids, rejected = self.detector_rect_markers.detectMarkers(frame)

        if ids is None:
            self.__exit()
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if len(corners) != 4:
            self.__exit()
            return

        centers = []
        marker_data = {}  # {id: {'center': (x, y), 'corners': [(x1,y1), ...], 'rvec': rvec, 'tvec': tvec}}

        for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
            center = np.mean(corner[0], axis=0)
            centers.append(center)

            marker_corners = corner[0]
            reordered_corners = list(reversed(marker_corners))
            tvec, rvec = self._estimate_marker_3d_pose(reordered_corners)

            marker_id_int = int(marker_id)
            smoothed_tvec = self._apply_smoothing(marker_id_int, tvec.squeeze())

            marker_data[marker_id_int] = {
                'main_corner': None,
                'corners': [tuple(map(float, c)) for c in reordered_corners],
                'tvec': smoothed_tvec,
                'rvec': rvec
            }

            cv2.circle(frame, tuple(center.astype(int)), 3, Config.COLORS['center'], -1)

        self._state.current_frame = frame
        self._state.src_points = np.float32(self._sort_points(centers, corners, ids, marker_data))
        self._state.marker_data = marker_data
        self._state.dvecs = self._calculate_diagonal_vector()
        self._state.start_vecs = (self._ids_diag[0], self._ids_diag[1])

    def _apply_smoothing(self, marker_id: int, new_tvec: np.ndarray) -> np.ndarray:
        """Применяет экспоненциальное сглаживание к вектору перемещения"""
        if marker_id not in self._smoothed_tvecs:
            self._smoothed_tvecs[marker_id] = new_tvec.copy()
            return new_tvec

        old_tvec = self._smoothed_tvecs[marker_id]
        smoothed = self._smoothing_alpha * new_tvec + (1 - self._smoothing_alpha) * old_tvec

        self._smoothed_tvecs[marker_id] = smoothed
        return smoothed

    def _estimate_marker_3d_pose(self, marker_corners_2d):
        """Оценивает 3D позицию и ориентацию маркера"""
        size = Config.MARKER_SIZE
        object_points = np.array([
            [-size / 2, -size / 2, 0],
            [size / 2, -size / 2, 0],
            [size / 2, size / 2, 0],
            [-size / 2, size / 2, 0]
        ], dtype=np.float32)

        marker_corners_2d = np.array(marker_corners_2d, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners_2d,
            Config.camera_matrix,
            Config.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success:
            return tvec, rvec
        else:
            self.__exit()

    def __exit(self):
        self._state.src_points = []
        self._state.method = Method.ERROR

    @staticmethod
    def _sort_points(points, corners, ids, data):
        """Сортирует точки в порядке: top-left, top-right, bottom-right, bottom-left
        Для каждого маркера сохраняет противоположный угол"""
        points = np.array(points)
        y_sorted_indices = np.argsort(points[:, 1])
        y_sorted = points[y_sorted_indices]

        top_points_indices = y_sorted_indices[:2]
        bottom_points_indices = y_sorted_indices[2:]

        top_points = y_sorted[:2]
        bottom_points = y_sorted[2:]

        top_sorted_order = np.argsort(top_points[:, 0])
        top_sorted_indices = top_points_indices[top_sorted_order]
        tl_idx, tr_idx = top_sorted_indices[0], top_sorted_indices[1]

        bottom_sorted_order = np.argsort(bottom_points[:, 0])
        bottom_sorted_indices = bottom_points_indices[bottom_sorted_order]
        bl_idx, br_idx = bottom_sorted_indices[0], bottom_sorted_indices[1]

        result_points = []

        for idx in [tl_idx, tr_idx, br_idx, bl_idx]:
            marker_corners = corners[idx][0]
            marker_id = int(ids[idx])

            corner_sums = marker_corners[:, 0] + marker_corners[:, 1]

            if idx == tl_idx:
                corner_point = marker_corners[np.argmax(corner_sums)]
            elif idx == tr_idx:
                corner_diffs = marker_corners[:, 0] - marker_corners[:, 1]
                corner_point = marker_corners[np.argmin(corner_diffs)]
            elif idx == br_idx:
                corner_point = marker_corners[np.argmin(corner_sums)]
            else:
                corner_diffs = marker_corners[:, 0] - marker_corners[:, 1]
                corner_point = marker_corners[np.argmax(corner_diffs)]

            data[marker_id]['main_corner'] = corner_point
            result_points.append(corner_point)

        return result_points


    def _calculate_diagonal_vector(self):
        """Вычисляет 3D вектор диагонали прямоугольника маркеров"""
        tl_2d, tr_2d, br_2d, bl_2d = self._state.src_points
        marker_data = self._state.marker_data

        if not self._ids_diag:
            self._ids_diag = list(repeat(None, 4))
            for marker_id, data in marker_data.items():
                marker_corners = data['corners']

                for corner in marker_corners:
                    corner = np.array(corner)
                    if np.allclose(corner, tl_2d, atol=1e-6):
                        self._ids_diag[0] = marker_id
                        continue

                    if np.allclose(corner, tr_2d, atol=1e-6):
                        self._ids_diag[1] = marker_id
                        continue

                    if np.allclose(corner, br_2d, atol=1e-6):
                        self._ids_diag[2] = marker_id
                        continue

                    if np.allclose(corner, bl_2d, atol=1e-6):
                        self._ids_diag[3] = marker_id
                        continue

        tl_3d = marker_data[self._ids_diag[0]]['tvec']
        tr_3d = marker_data[self._ids_diag[1]]['tvec']
        br_3d = marker_data[self._ids_diag[2]]['tvec']
        bl_3d = marker_data[self._ids_diag[3]]['tvec']

        return br_3d - tl_3d, bl_3d - tr_3d


    def reset(self):
        self._ids_diag = []
        self._smoothed_tvecs = {}