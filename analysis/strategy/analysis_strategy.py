import numpy as np

class EmptyAnalysisStrategy:
    """Никак не обрабатывает картинку. Для теста браузера"""
    def __call__(self, frame:np.ndarray)->np.ndarray:
        return frame