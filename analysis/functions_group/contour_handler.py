from typing import Optional

import numpy as np

from analysis.analysis_config import Config
from analysis.analysis_state import State, Method
from analysis.functions.contour_part.find_contour import FindContour
from analysis.functions.contour_part.process_contour import ProcessContour
from analysis.functions_group.functions_group import FunctionsGroup


class ContourHandler(FunctionsGroup):
    def __init__(self, state: State):
        super().__init__(state)
        self._STARTED_METHOD = Method.FIND_CONTOUR
        self._transition = {
            Method.FIND_CONTOUR:    (Method.PROCESS_CONTOUR, FindContour(self._state)),
            Method.PROCESS_CONTOUR: (Method.EXIT, ProcessContour(self._state))
        }

        self._prev_dvec:Optional[np.ndarray] = None
        self._cur_dvec:Optional[np.ndarray] = None
        self._sum_angle = 0

    def __call__(self, *args, **kwargs):
        """Обрабатывает текущий кадр и запускает анализ контура при необходимости."""
        self._cur_dvec = self._state.dvecs[0]

        if self._prev_dvec is None:
            self._prev_dvec = self._cur_dvec
            super().__call__(*args, **kwargs)
            return

        norm1 = float(np.linalg.norm(self._cur_dvec))
        norm2 = float(np.linalg.norm(self._prev_dvec))

        angle = np.arccos((self._cur_dvec @ self._prev_dvec) / (norm1 * norm2))

        if angle > np.pi / Config.PHOTO_COUNTS:
            self._prev_dvec = self._cur_dvec
            self._sum_angle += angle
            super().__call__(*args, **kwargs)
            return

        self._state.method = Method.EXIT

    def reset(self):
        """Подготовка к переиспользованию класса"""
        self._sum_angle = 0
        self._prev_dvec = None
        self._cur_dvec = None

        for _, (_, method) in self._transition.items():
            method.reset()

    @property
    def sum_angle(self):
        return self._sum_angle