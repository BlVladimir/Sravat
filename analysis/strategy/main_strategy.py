from analysis.analysis_state import State, Method

import numpy as np
from logging import getLogger

from analysis.functions_group.contour_handler import ContourHandler
from analysis.functions_group.markers_handler import MarkersHandler


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self._state = State()

        self._logger = getLogger(type(self).__name__)

        self._markers_handler = MarkersHandler(self._state)
        self._contour_handler = ContourHandler(self._state)

    def __call__(self, frame:np.ndarray)->np.ndarray:
        self._state.current_frame = frame.copy()

        self._markers_handler()
        if self._state.method == Method.ERROR:
            return frame

        self._markers_handler()
        if self._state.method == Method.ERROR:
            return frame

        return self._state.current_frame