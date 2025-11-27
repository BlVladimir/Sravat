from typing import Protocol

import numpy as np


class AnalysisStrategyInterface(Protocol):
    """Интерфейс стратегии обработки картинки. Использовать в проверках на тип"""
    def __call__(self, frame:np.ndarray)->np.ndarray:
        """
        Использует объект как функцию для обработки.
        base64_input - строковое представление Base64.
        Возвращает строковое (временно) представление Base64.
        """
        pass

class EmptyAnalysisStrategy:
    """Никак не обрабатывает картинку. Для теста браузера"""
    def __call__(self, frame:np.ndarray)->np.ndarray:
        return frame