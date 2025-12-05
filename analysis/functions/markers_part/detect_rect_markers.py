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
            # Центр маркера
            center = np.mean(corner[0], axis=0)
            centers.append(center)

            marker_corners = corner[0]
            reordered_corners = list(reversed(marker_corners))
            tvec, rvec = self._estimate_marker_3d_pose(reordered_corners)

            marker_data[int(marker_id)] = {
                'center': tuple(center),
                'corners': [tuple(map(float, c)) for c in reordered_corners],
                'tvec': tvec.squeeze(),
                'rvec': rvec
            }

            cv2.circle(frame, tuple(center.astype(int)), 3, Config.COLORS['center'], -1)

        self._state.current_frame = frame
        self._state.src_points = np.float32(self._sort_points(centers))
        self._state.marker_data = marker_data
        self._state.dvecs = self._calculate_diagonal_vector()
        self._state.start_vecs = (self._ids_diag[1], self._ids_diag[0])

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

        # Решаем задачу PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners_2d,
            Config.camera_matrix,
            Config.dist_coeffs
        )

        if success:
            return tvec, rvec  # вектор вращения и вектор перемещения
        else:
            self._logger.warning("Can't estimate marker pose")
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def __exit(self):
        self._state.src_points = []
        self._state.method = Method.END

    @staticmethod
    def _sort_points(points):
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

    def _calculate_diagonal_vector(self):
        """Вычисляет 3D вектор диагонали прямоугольника маркеров"""
        tl_2d, tr_2d, br_2d, bl2d = self._state.src_points
        marker_data = self._state.marker_data

        if not self._ids_diag:
            for marker_id, data in marker_data.items():
                marker_center = np.array(data['center'], dtype=np.float32)

                if np.linalg.norm(marker_center - tl_2d) < 1.0:
                    tl_3d = data['tvec']
                    self._ids_diag.append(marker_id)

                if np.linalg.norm(marker_center - br_2d) < 1.0:
                    br_3d = data['tvec']
                    self._ids_diag.append(marker_id)

                if np.linalg.norm(marker_center - tr_2d) < 1.0:
                    tr_3d = data['tvec']
                    self._ids_diag.append(marker_id)

                if np.linalg.norm(marker_center - bl2d) < 1.0:
                    bl_3d = data['tvec']
                    self._ids_diag.append(marker_id)
        else:
            tl_3d = marker_data[self._ids_diag[0]]['tvec']
            tr_3d = marker_data[self._ids_diag[1]]['tvec']
            br_3d = marker_data[self._ids_diag[2]]['tvec']
            bl_3d = marker_data[self._ids_diag[3]]['tvec']

        return br_3d - tr_3d, bl_3d - tl_3d

    def reset(self):
        self._ids_diag = []