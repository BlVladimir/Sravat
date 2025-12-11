import logging
from logging import debug

import numpy as np

from analysis.analysis_config import Config
from analysis.strategy.camera_calibration_strategy import CameraCalibrationStrategy
from analysis.strategy.main_strategy import MainAnalysisStrategy


class FacadeAnalysis:
    """
    Изоляция системы взаимодействий
    """
    def __init__(self):
        Config.load_config()
        self._camera_calibration = CameraCalibrationStrategy()
        self._is_calibrated = Config.load_calibration()

        self._main_strategy = MainAnalysisStrategy()

    def analyze_frame(self, frame:np.ndarray)->np.ndarray:
        if self._is_calibrated:
            return self._main_strategy(frame)
        else:
            frame = self._camera_calibration(frame)
            self._is_calibrated = Config.load_calibration()
            return frame

    def recalibrate(self):
        self._camera_calibration.reset()
        self._is_calibrated = False

    def reset(self):
        self._main_strategy.reset()


class EmptyFacadeAnalysis:
    """Тестировачный пустой фасад"""
    def __init__(self):
        logging.warn('Фасад создан')

    def analyze_frame(self, frame:np.ndarray)->np.ndarray:
        return frame