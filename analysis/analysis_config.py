import os

import numpy as np
from logging import error, warning


class Config:
    """Важные константы программы"""
    COLORS = {
        'contour': (255, 0, 0),  # Синий контур
        'fill': (0, 255, 255),  # Желтая заливка
        'center': (0, 255, 0),  # Зеленый центр
        'corners': [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Угловые точки
    }

    MARKER_SIZE = 0.05  # Физический размер маркеров в м.
    PHOTO_COUNTS = 1  # Количество данных для создания модели (для 1 просто призма контура. >1 если бы переход к 3D работал стабильно, улучшало бы качество 3D модели)
    EDGE = 200  # Максимальное количество вокселей по одной из осей

    ANGLE_OF_INCIDENCE = np.pi / 4  # угол падения света

    camera_matrix = None  # Искажения камер. Считается автоматически
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
            error(f'Ошибка загрузки калибровки: {e}')
            return False