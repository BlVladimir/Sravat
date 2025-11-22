import base64
from logging import getLogger
from typing import Protocol
import cv2
import numpy as np

from analysis.analysis_environment import Environment
from analysis.functions.create_homography_transform import CreateHomographyTransform
from analysis.functions.detect_markers import DetectMarkers
from analysis.functions.draw_plane import DrawPlane


class AnalysisStrategyInterface(Protocol):
    """Интерфейс стратегии обработки картинки. Использовать в проверках на тип"""
    def __call__(self, base64_input:str)->str:
        """
        Использует объект как функцию для обработки.
        base64_input - строковое представление Base64.
        Возвращает строковое (временно) представление Base64.
        """
        pass

class EmptyAnalysisStrategy:
    """Никак не обрабатывает картинку. Для теста браузера"""
    def __call__(self, base64_input:str)->str:
        return base64_input

class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self.environment = Environment()

        self.environment.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.environment.aruco_params = cv2.aruco.DetectorParameters()
        self.environment.detector = cv2.aruco.ArucoDetector(self.environment.aruco_dict, self.environment.aruco_params)


        self.logger = getLogger(type(self).__name__)

        self.pipeline = [
            DetectMarkers(self.environment),
            CreateHomographyTransform(self.environment),
            DrawPlane(self.environment),
        ]

    def __call__(self, base64_input:str)->str:
        try:
            frame = self.to_cv2(base64_input)

            self.environment.current_frame = frame

            for function in self.pipeline:
                function()

            frame = self.environment.current_frame

            result_base64 = self.to_base64(frame)

            return result_base64
        except Exception as e:
            self.logger.error(f'Error in main analysis: {e}')
            return ""

    def to_cv2(self, base64_string: str):
        """Конвертирует Base64 строку в изображение OpenCV"""
        try:
            img_data = base64.b64decode(base64_string)

            np_array = np.frombuffer(img_data, np.uint8)

            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError('Failed to decode image from Base64')

            return image
        except Exception as e:
            self.logger.error(f'Error converting Base64 to OpenCV: {e}')
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def to_base64(self, image: np.ndarray):
        """Конвертирует изображение OpenCV в Base64 строку"""
        try:
            success, encoded_image = cv2.imencode('.jpg', image)

            if not success:
                raise ValueError('Failed to encode image to JPEG')

            base64_string = base64.b64encode(encoded_image).decode('utf-8')

            return base64_string
        except Exception as e:
            self.logger.error(f'Error converting OpenCV to Base64: {e}')
            return ''