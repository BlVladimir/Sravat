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
        self._idx_corners = {}


    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self._state.current_frame
        corners, ids = self._detect_markers_subpixel(frame)

        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        centers = []
        marker_data = {}

        for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
            center = np.mean(corner[0], axis=0)
            centers.append(center)

            marker_corners = corner[0]
            reordered_corners = list(reversed(marker_corners))
            tvec, rvec, corners_3d = self._estimate_marker_3d_pose(reordered_corners)

            marker_data[int(marker_id)] = {
                'main_corner': None,
                'corners': [tuple(map(float, c)) for c in reordered_corners],
                'corners_3d': corners_3d,
                'tvec': None,
                'rvec': rvec
            }

            cv2.circle(frame, tuple(center.astype(int)), 3, Config.COLORS['center'], -1)

        self._state.current_frame = frame
        self._state.src_points = np.float32(self._sort_points(centers, corners, ids, marker_data))
        self._state.marker_data = marker_data
        self._state.dvecs = self._calculate_diagonal_vector()
        self._state.start_vecs = (self._ids_diag[0], self._ids_diag[1])

    def _detect_markers_subpixel(self, frame):
        """Детектирует маркеры с субпиксельной точностью"""
        corners, ids, rejected = self.detector_rect_markers.detectMarkers(frame)

        if ids is None or len(corners) != 4:
            self.__exit()
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        refined_corners = []
        for corner in corners:
            corner_reshaped = corner.reshape(-1, 1, 2).astype(np.float32)
            cv2.cornerSubPix(gray, corner_reshaped, (3, 3), (-1, -1), criteria)
            refined_corners.append(corner_reshaped.reshape(1, 4, 2))

        return refined_corners, ids

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
            Config.dist_coeffs
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            corners_3d = {}
            for i, corner_local in enumerate(object_points):
                corner_world = rotation_matrix @ corner_local + tvec.squeeze()
                corners_3d[i] = corner_world

            return tvec, rvec, corners_3d
        else:
            self.__exit()
            return None, None, None

    def __exit(self):
        self._state.src_points = []
        self._state.method = Method.ERROR

    def _sort_points(self, points, corners, ids, data):
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
            if marker_id not in self._idx_corners:
                corner_sums = marker_corners[:, 0] + marker_corners[:, 1]
                corner_diffs = marker_corners[:, 0] - marker_corners[:, 1]

                if idx == tl_idx:
                    corner_idx = np.argmax(corner_sums)
                elif idx == tr_idx:
                    corner_idx = np.argmin(corner_diffs)
                elif idx == br_idx:
                    corner_idx = np.argmin(corner_sums)
                else:
                    corner_idx = np.argmax(corner_diffs)

                self._idx_corners[marker_id] = corner_idx
            else:
                corner_idx = self._idx_corners[marker_id]

            corner_point = marker_corners[corner_idx]
            data[marker_id]['main_corner'] = corner_point

            corners_3d = data[marker_id]['corners_3d']
            original_corner_idx = 3 - corner_idx
            data[marker_id]['tvec'] = corners_3d[original_corner_idx]

            result_points.append(corner_point)

        return result_points

    def _calculate_diagonal_vector(self):
        """Вычисляет 3D вектор диагонали прямоугольника маркеров"""
        tl_2d, tr_2d, br_2d, bl_2d = self._state.src_points
        marker_data = self._state.marker_data

        if not self._ids_diag:
            self._logger.info("=== Initializing diagonal IDs (first time) ===")
            self._ids_diag = list(repeat(None, 4))
            for marker_id, data in marker_data.items():
                marker_corners = data['corners']
                self._logger.debug(f"Checking marker {marker_id}")

                for corner_idx, corner in enumerate(marker_corners):
                    corner = np.array(corner)
                    if np.allclose(corner, tl_2d, atol=1e-6):
                        self._logger.info(f"  TL matched: marker_id={marker_id}, corner_idx={corner_idx}, 2D={corner}")
                        self._ids_diag[0] = marker_id
                        continue

                    if np.allclose(corner, tr_2d, atol=1e-6):
                        self._logger.info(f"  TR matched: marker_id={marker_id}, corner_idx={corner_idx}, 2D={corner}")
                        self._ids_diag[1] = marker_id
                        continue

                    if np.allclose(corner, br_2d, atol=1e-6):
                        self._logger.info(f"  BR matched: marker_id={marker_id}, corner_idx={corner_idx}, 2D={corner}")
                        self._ids_diag[2] = marker_id
                        continue

                    if np.allclose(corner, bl_2d, atol=1e-6):
                        self._logger.info(f"  BL matched: marker_id={marker_id}, corner_idx={corner_idx}, 2D={corner}")
                        self._ids_diag[3] = marker_id
                        continue

        self._logger.info(f"Diagonal IDs: TL={self._ids_diag[0]}, TR={self._ids_diag[1]}, BR={self._ids_diag[2]}, BL={self._ids_diag[3]}")

        tl_3d = marker_data[self._ids_diag[0]]['tvec']
        tr_3d = marker_data[self._ids_diag[1]]['tvec']
        br_3d = marker_data[self._ids_diag[2]]['tvec']
        bl_3d = marker_data[self._ids_diag[3]]['tvec']

        diag_main = br_3d - tl_3d
        diag_aux = bl_3d - tr_3d


        norm1 = float(np.linalg.norm(diag_main))
        norm2 = float(np.linalg.norm(diag_aux))

        cos = np.clip((diag_main @ diag_aux) / (norm1 * norm2), -1, 1)

        angle = np.arccos(cos)

        self._logger.info(f'Angle between diagonals: {angle:.2f} deg, ratio: {norm1/norm2:.2f}.')  # Главная проблема: по логам видно, что инвариантные независимо от базиса характеристики плоскости различны у разных ракурсов

        return diag_main, diag_aux


    def reset(self):
            self._ids_diag = []
            self._idx_corners = {}