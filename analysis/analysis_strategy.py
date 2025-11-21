from typing import Protocol
import cv2

from analysis.analysis_environment import Environment
from analysis.function import FindRect, QuiteFunction


class AnalysisStrategyInterface(Protocol):
    """Интерфейс стратегии обработки картинки. Использовать в проверках на тип"""
    def __call__(self, data:str)->str:
        """
            Использует объект как функцию для обработки.
            data - строковое (временно) представление Base64.
            Возвращает строковое (временно) представление Base64.
        """
        pass

class EmptyAnalysisStrategy:
    """Никак не обрабатывает картинку"""
    def __call__(self, data:str)->str:
        return data

class MainAnalysisStrategy:
    """Обычный алгоритм"""
    def __init__(self):
        self.environment = Environment()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        self.environment.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)  # вынести в dataclass
        self.environment.cap = cv2.VideoCapture(0)

        self.find_rect = FindRect(self.environment)
        self.quite = QuiteFunction(self.environment)

    def __call__(self, data:str)->str:
        self.find_rect()
        if not self.quite():
            return data
        else:
            return ''  # костыль для реализации выхода