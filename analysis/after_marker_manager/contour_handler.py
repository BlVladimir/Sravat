from logging import getLogger
from typing import Optional

import numpy as np

from analysis.analysis_config import Config
from analysis.analysis_state import State, Method
from analysis.functions.contour_part.find_contour import FindContour
from analysis.functions.contour_part.process_contour import ProcessContour


class ContourHandler:
    """Обрабатывает контуры для создания 3D"""
    def __init__(self, state:State):
        self._state:State = state
        self._logger = getLogger(type(self).__name__)
        self._prev_dvec:Optional[np.ndarray] = None
        self._cur_dvec:Optional[np.ndarray] = None
        self._sum_angle = 0

        self._transition = {
            Method.END_MARKER_PART: (Method.FIND_CONTOUR, lambda:None),
            Method.FIND_CONTOUR:    (Method.PROCESS_CONTOUR, FindContour(self._state)),
            Method.PROCESS_CONTOUR: (Method.END, ProcessContour(self._state))
        }  # переходы между состояниями

    def process_frame(self):
        """Обрабатывает текущий кадр и запускает анализ контура при необходимости."""

        self._cur_dvec = self._state.dvecs[0]

        if self._prev_dvec is None:
            self._prev_dvec = self._cur_dvec
            self._process()
            return

        norm1 = float(np.linalg.norm(self._cur_dvec))
        norm2 = float(np.linalg.norm(self._prev_dvec))

        angle = np.arccos((self._cur_dvec @ self._prev_dvec) / (norm1 * norm2))
        self._logger.info(angle)

        if angle > np.pi / Config.PHOTO_COUNTS:
            self._prev_dvec = self._cur_dvec
            self._sum_angle += angle
            self._process()
            return

        self._state.method = Method.END

    def reset(self):
        """Подготовка к переиспользованию класса"""
        self._sum_angle = 0
        self._prev_dvec = None
        self._cur_dvec = None

        for _, (_, method) in self._transition.items():
            method.reset()

    def _process(self):
        """Обрабатывает контур"""
        self._logger.info('Processing contour...')
        while self._state.method != Method.END and self._state.method != Method.ERROR:
            next_method, method = self._transition[ self._state.method]
            self._state.method = next_method
            method()

    @property
    def sum_angle(self):
        return self._sum_angle