import os

import numpy as np
from logging import error, warning


class Config:
    colors = {
        'contour': (255, 0, 0),  # Синий контур
        'fill': (0, 255, 255),  # Желтая заливка
        'center': (0, 255, 0),  # Зеленый центр
        'corners': [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Угловые точки
    }

    marker_size = 0.05

    camera_matrix = None
    dist_coeffs = None

    @classmethod
    def load_calibration(cls):
        """Загружает результаты калибровки"""
        if not os.path.exists('camera_calibration.npz'):
            warning(f"Файл калибровки не найден")
            return False
        try:
            data = np.load('camera_calibration.npz')
            cls.camera_matrix = data['camera_matrix']
            cls.dist_coeffs = data['dist_coeffs']
            return True
        except Exception as e:
            error(f"Ошибка загрузки калибровки: {e}")
            return False