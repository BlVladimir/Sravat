import numpy as np

from analysis.analysis_config import Config
from analysis.strategy.analysis_strategy import AnalysisStrategyInterface
from analysis.strategy.camera_calibration_strategy import CameraCalibrationStrategy


class FacadeAnalysis:
    """Класс, через который осуществляется взаимодействие обработки с сайтом
    Данный класс создан 24.11.25. Без намеков
    """
    def __init__(self, strategy:AnalysisStrategyInterface):
        self._camera_calibration = CameraCalibrationStrategy()
        self._is_calibrated = Config.load_calibration()

        self._strategy = strategy

    def analyze_frame(self, frame:np.ndarray)->np.ndarray:
        if self._is_calibrated:
            return self._strategy(frame)
        else:
            frame = self._camera_calibration(frame)
            self._is_calibrated = Config.load_calibration()
            return frame