import numpy as np

from analysis.analysis_config import Config
from analysis.strategy.analysis_strategy import AnalysisStrategyInterface
from analysis.strategy.camera_calibration_strategy import CameraCalibrationStrategy


class FacadeAnalysis:
    """Класс, через который осуществляется взаимодействие обработки с сайтом
    Данный класс создан 24.11.25. Без намеков
    """
    def __init__(self, strategy:AnalysisStrategyInterface):
        self.camera_calibration = CameraCalibrationStrategy()
        self.is_calibrated = Config.load_calibration()

        self.strategy = strategy

    def analyze_frame(self, frame:np.ndarray)->np.ndarray:
        if self.is_calibrated:
            return self.strategy(frame)
        else:
            frame = self.camera_calibration(frame)
            self.is_calibrated = self.camera_calibration.is_calibrated
            return frame