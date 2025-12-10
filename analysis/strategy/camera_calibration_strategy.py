from logging import getLogger

import numpy as np
import cv2

from analysis.analysis_config import Config


class CameraCalibrationStrategy:
    """Стратегия калибровки камеры с использованием шахматной доски"""

    def __init__(self):
        self._logger = getLogger(type(self).__name__)
        self._NUM_IMAGES = 20

        self._chessboard_size = (9, 6)
        self._square_size = 0.02

        self._obj_points = np.zeros((self._chessboard_size[0] * self._chessboard_size[1], 3), np.float32)
        self._obj_points[:, :2] = np.mgrid[0:self._chessboard_size[0], 0:self._chessboard_size[1]].T.reshape(-1, 2)
        self._obj_points *= self._square_size

        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self._all_obj_points = []
        self._all_img_points = []
        self._image_size = None
        self._captured_count = 0

        self._camera_matrix = None
        self._dist_coeffs = None
        self.is_calibrated = False

        self._filename = 'camera_calibration.npz'

        self.prev_frame = None
        self.change_threshold = 0.15

        self.enable_change_detection = True

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Обрабатывает кадр с учетом изменений относительно предыдущего кадра.
        """
        if self.is_calibrated:
            return frame

        self._image_size = (frame.shape[1], frame.shape[0])

        if not self._should_process_frame(frame):
            return frame

        self._process_frame(frame)

        self.prev_frame = frame
        return frame

    def _should_process_frame(self, frame: np.ndarray) -> bool:
        """
        Определяет, нужно ли обрабатывать текущий кадр.
        """
        if self.prev_frame is None:
            return True

        if not self.enable_change_detection:
            return True

        change_percentage = self._calculate_frame_change(self.prev_frame, frame)

        return change_percentage > self.change_threshold

    def _calculate_frame_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Вычисляет процент изменения между двумя кадрами.
        """
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray1, gray2)

            _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            changed_pixels = np.count_nonzero(thresholded)
            total_pixels = gray1.size

            change_percentage = changed_pixels / total_pixels

            return change_percentage

        except Exception as e:
            self._logger.error(f"Ошибка при вычислении изменения кадра: {e}")
            return 1.0

    def _process_frame(self, frame: np.ndarray):
        """
        Основная обработка кадра: поиск шахматной доски и сохранение данных.
        """
        # Преобразуем в grayscale для поиска углов
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        found, corners = cv2.findChessboardCorners(
            gray,
            self._chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                self._criteria
            )

            self._all_obj_points.append(self._obj_points)
            self._all_img_points.append(corners_refined)
            self._captured_count += 1


            if self._captured_count >= self._NUM_IMAGES and not self.is_calibrated:
                self._calibrate()

    def _calibrate(self):
        """Выполняет калибровку камеры"""
        try:
            self._logger.info("Начинаю калибровку камеры...")

            ret, self._camera_matrix, self._dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self._all_obj_points,
                self._all_img_points,
                self._image_size,
                None,
                None,
                flags=cv2.CALIB_FIX_K3
            )

            self.is_calibrated = True


            self._save_calibration()

            Config.update_calibration(self._camera_matrix, self._dist_coeffs)

        except Exception as e:
            self._logger.error(f'Ошибка калибровки: {e}')

    def _save_calibration(self):
        """Сохраняет результаты калибровки в файл"""
        np.savez(self._filename,
                 camera_matrix=self._camera_matrix,
                 dist_coeffs=self._dist_coeffs,
                 image_size=self._image_size,
                 chessboard_size=self._chessboard_size,
                 square_size=self._square_size,
                 calibration_date=np.datetime64('now'))

        self._logger.info(f"Калибровка сохранена в {self._filename}")

    def reset(self):
        """Сбрасывает данные калибровки для новой попытки"""
        self._all_obj_points = []
        self._all_img_points = []
        self._captured_count = 0
        self.is_calibrated = False
        self._camera_matrix = None
        self._dist_coeffs = None
        self._logger.info("Данные калибровки сброшены")