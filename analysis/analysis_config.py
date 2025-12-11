import os
import json

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

    MARKER_RECT_SIZE = 0.05  # Физический размер маркеров четырехугольника в м.
    MARKER_LIGHT_SIZE = 0.08  # Физический размер маркера света в м.
    PHOTO_COUNTS = 20  # Количество данных для создания модели (для 1 просто призма контура. >1 если бы переход к 3D работал стабильно, улучшало бы качество 3D модели)
    EDGE = 100  # Максимальное количество вокселей по одной из осей

    ANGLE_OF_INCIDENCE = 45  # угол падения света

    camera_matrix = None  # Искажения камер. Считается автоматически
    dist_coeffs = None

    @classmethod
    def load_config(cls):
        """Загружает конфигурацию из файла config.json"""
        config_file = 'config.json'
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                cls.MARKER_RECT_SIZE = config_dict.get('MARKER_RECT_SIZE', cls.MARKER_RECT_SIZE)
                cls.MARKER_LIGHT_SIZE = config_dict.get('MARKER_LIGHT_SIZE', cls.MARKER_LIGHT_SIZE)
                cls.PHOTO_COUNTS = config_dict.get('PHOTO_COUNTS', cls.PHOTO_COUNTS)
                cls.EDGE = config_dict.get('EDGE', cls.EDGE)
                cls.ANGLE_OF_INCIDENCE = config_dict.get('ANGLE_OF_INCIDENCE', cls.ANGLE_OF_INCIDENCE)
            else:
                cls._save_config()
        except Exception as e:
            error(f'Ошибка загрузки конфигурации: {e}. Создаю новый файл конфигурации.')
            cls._save_config()

    @classmethod
    def _save_config(cls):
        """Сохраняет текущую конфигурацию в файл config.json"""
        config_dict = {
            'MARKER_RECT_SIZE': cls.MARKER_RECT_SIZE,
            'MARKER_LIGHT_SIZE': cls.MARKER_LIGHT_SIZE,
            'PHOTO_COUNTS': cls.PHOTO_COUNTS,
            'EDGE': cls.EDGE,
            'ANGLE_OF_INCIDENCE': float(cls.ANGLE_OF_INCIDENCE)
        }

        try:
            with open('config.json', 'w') as f:
                json.dump(config_dict, f, indent=4)
        except Exception as e:
            error(f'Ошибка сохранения конфигурации: {e}')

    @classmethod
    def load_calibration(cls):
        """Загружает результаты калибровки"""
        if not os.path.exists('camera_calibration.npz'):
            return False
        try:
            data = np.load('camera_calibration.npz')
            cls.camera_matrix = data['camera_matrix']
            cls.dist_coeffs = data['dist_coeffs']
            return True
        except Exception as e:
            error(f'Ошибка загрузки калибровки: {e}')
            return False

    @classmethod
    def update_calibration(cls, camera_matrix, dist_coeffs):
        """Обновляет параметры калибровки камеры"""
        cls.camera_matrix = camera_matrix
        cls.dist_coeffs = dist_coeffs