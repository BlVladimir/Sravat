from analysis.analysis_config import Config
from analysis.analysis_state import State, Method

import numpy as np
from logging import getLogger

from analysis.functions_group.contour_handler import ContourHandler
from analysis.functions_group.markers_handler import MarkersHandler
from analysis.functions_group.process_data import ProcessData
from analysis.functions_group.shadow_handler import ShadowHandler


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self._state = State()

        self._logger = getLogger(type(self).__name__)

        self._markers_handler = MarkersHandler(self._state)
        self._contour_handler = ContourHandler(self._state)
        self._process_data = ProcessData(self._state)
        self._shadow_handler = ShadowHandler(self._state)

        self._is_processed = False

    def __call__(self, frame:np.ndarray)->np.ndarray:
        self._state.current_frame = frame.copy()

        self._markers_handler()
        if self._state.method == Method.ERROR:
            return frame

        # self._contour_handler()
        # if self._state.method == Method.ERROR:
        #     return frame

        if not self._is_processed:
            self._contour_handler()
            if self._state.method == Method.ERROR:
                return frame

            if len(self._state.scanning_data) == Config.PHOTO_COUNTS:
                self._process_data()
                self._is_processed = True

        return self._state.current_frame