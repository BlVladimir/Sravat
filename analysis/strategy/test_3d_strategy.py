from itertools import repeat
from logging import getLogger

import numpy as np

from analysis.analysis_state import State, Method
from analysis.functions_group.markers_handler import MarkersHandler
from analysis.functions_group.process_data import ProcessData


class Test3DStrategy:
    """Тестировочная стратегия"""
    def __init__(self):
        self._state = State()

        self._logger = getLogger(type(self).__name__)

        self._markers_handler = MarkersHandler(self._state)

        data = (np.array([3, 0, 0]),
                np.array([0, 4, 0]),
                np.array([0, 2, 0]),
                np.array([1, 1, 0]),
                np.array([[100,
                           np.sin(n * 2 * np.pi / 360),
                           np.cos(n * 2 * np.pi / 360)]
                          for n in range(0, 360)], dtype=np.float32))

        scanning_data = list(repeat(data, 20))

        self._state.scanning_data = scanning_data

        self._process_data = ProcessData(self._state)
        self._is_processed = False

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        self._state.current_frame = frame.copy()

        self._markers_handler()
        if self._state.method == Method.ERROR:
            return frame

        if self._is_processed:
            return self._state.current_frame
        else:
            self._process_data()
            self._is_processed = True
            return self._state.current_frame