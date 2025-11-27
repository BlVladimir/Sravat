import numpy as np

from analysis.strategy.analysis_strategy import AnalysisStrategyInterface


class FacadeAnalysis:
    """Класс, через который осуществляется взаимодействие обработки с сайтом
    Данный класс создан 24.11.25. Без намеков
    """
    def __init__(self, strategy:AnalysisStrategyInterface):
        self.strategy = strategy

    def analyze_frame(self, frame:np.ndarray)->np.ndarray:
        return self.strategy(frame)