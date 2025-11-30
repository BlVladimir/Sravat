from analysis.analysis_config import Config
from analysis.analysis_state import State
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

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self.state.current_frame
        corners, ids, rejected = self.detector_rect_markers.detectMarkers(frame)

        if ids is None:
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        centers = []
        marker_data = {}  # {id: {'center': (x, y), 'corners': [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]}}

        for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
            # Центр маркера
            center = np.mean(corner[0], axis=0)
            centers.append(center)

            marker_corners = corner[0]
            reordered_corners = list(reversed(marker_corners))

            marker_data[int(marker_id)] = {
                'center': tuple(center),
                'corners': [tuple(map(float, c)) for c in reordered_corners]
            }

            cv2.circle(frame, tuple(center.astype(int)), 3, Config.colors['center'], -1)

            # Рисуем нумерацию углов для визуализации (опционально)
            for idx, corner_point in enumerate(reordered_corners):
                color = Config.colors['corners'][idx] if idx < len(Config.colors['corners']) else (255, 255, 255)
                corner_int = tuple(corner_point.astype(int).tolist())
                cv2.circle(frame, corner_int, 5, color, -1)
                cv2.putText(frame, str(idx), corner_int,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # for corner in corners:
        #     center = np.mean(corner[0], axis=0)
        #     centers.append(center)
        #     # Рисуем центр маркера
        #     cv2.circle(frame, tuple(center.astype(int)), 3, Config.colors['center'], -1)

        self.state.current_frame = frame
        self.state.centers = centers
        self.state.marker_data = marker_data

    @staticmethod
    def estimate_marker_3d_pose(marker_corners_2d, camera_matrix, dist_coeffs):
        """
        Оценивает 3D позицию и ориентацию маркера

        Args:
            marker_corners_2d: 2D координаты углов маркера (4 точки)
            camera_matrix: матрица внутренних параметров камеры
            dist_coeffs: коэффициенты дисторсии
        """

        # Маркер лежит в плоскости Z=0
        size = Config.marker_size
        object_points = np.array([
            [-size / 2, -size / 2, 0],
            [size / 2, -size / 2, 0],
            [size / 2, size / 2, 0],
            [-size / 2, size / 2, 0]
        ], dtype=np.float32)

        # Решаем задачу PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners_2d,
            camera_matrix,
            dist_coeffs
        )

        if success:
            return rvec, tvec  # вектор вращения и вектор перемещения
        else:
            return None, None