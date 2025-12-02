from logging import getLogger

import numpy as np
import cv2

from analysis.analysis_config import Config


class CameraCalibrationStrategy:
    """Стратегия, получающая данные о камере"""
    def __init__(self):
        self.logger = getLogger(type(self).__name__)
        self.NUM_IMAGES = 5

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.all_obj_points = []  # 3D точки
        self.all_img_points = []  # 2D точки
        self.image_size = None
        self.captured_count = 0

        # Результаты калибровки
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_calibrated = False

        self.size = Config.marker_size
        self.marker_3d_points = np.array([
            [-self.size / 2, -self.size / 2, 0],
            [self.size / 2, -self.size / 2, 0],
            [self.size / 2, self.size / 2, 0],
            [-self.size / 2, self.size / 2, 0]
        ], dtype=np.float32)

        self.filename = 'camera_calibration.npz'

    def __call__(self, frame:np.ndarray)->np.ndarray:
        self.image_size = (frame.shape[1], frame.shape[0])

        corners, ids, rejected = self.detector.detectMarkers(frame)

        if ids is None or len(ids) == 0:
            return frame

        # Берем первый обнаруженный маркер
        for marker_corner in corners:
            reordered_corners = np.array(list(reversed(marker_corner[0])), dtype=np.float32)
            # Сохраняем данные
            obj_points = self.marker_3d_points.reshape(-1, 1, 3)

            self.all_obj_points.append(obj_points)
            self.all_img_points.append(reordered_corners.reshape(-1, 1, 2))
        self.captured_count += 1

        if self.captured_count >= self.NUM_IMAGES:
            self._calibrate()

        return frame

    def _calibrate(self):
        try:
            # Выполняем калибровку
            all_obj_points = [np.array(pts, dtype=np.float32).reshape(-1, 1, 3) for pts in self.all_obj_points]
            all_img_points = [np.array(pts, dtype=np.float32).reshape(-1, 1, 2) for pts in self.all_img_points]

            # Используем пустые массивы вместо None
            camera_matrix_init = np.zeros((3, 3), dtype=np.float32)
            dist_coeffs_init = np.zeros((5, 1), dtype=np.float32)

            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_points,
                all_img_points,
                self.image_size,
                camera_matrix_init,  # Вместо None
                dist_coeffs_init  # Вместо None
            )

            self.is_calibrated = True

            self._save_calibration()

        except Exception as e:
            self.logger.error(f"Ошибка калибровки: {e}")

    def _save_calibration(self):
        """Сохраняет результаты калибровки"""
        np.savez(self.filename,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 marker_length=self.size)