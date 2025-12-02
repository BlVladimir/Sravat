from analysis.analysis_state import State, Method

import numpy as np
from logging import getLogger

from analysis.functions.adaptive import Adaptive
from analysis.functions.canny import CannyMethod
from analysis.functions.create_homography_transform import CreateHomographyTransform
from analysis.functions.detect_light_marker import DetectLightMarker
from analysis.functions.detect_rect_markers import DetectRectMarkers
from analysis.functions.draw_plane import DrawPlane
from analysis.functions.find_contour import FindContour
from analysis.functions.process_contour import ProcessContour
from analysis.functions.select_detect_contour_method import SelectDetectContourMethod


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self._state = State()

        self.__logger = getLogger(type(self).__name__)

        self._transition = {
            Method.DETECT_RECT_MARKERS:         (Method.CREATE_HOMOGRAPHY_TRANSFORM, DetectRectMarkers(self._state)),
            Method.CREATE_HOMOGRAPHY_TRANSFORM: (Method.DETECT_LIGHT_MARKER, CreateHomographyTransform(self._state)),
            Method.DETECT_LIGHT_MARKER:         (Method.DRAW_PLANE, DetectLightMarker(self._state)),
            Method.DRAW_PLANE:                  (Method.FIND_CONTOUR, DrawPlane(self._state)),
            Method.FIND_CONTOUR:                (Method.PROCESS_CONTOUR, FindContour(self._state)),
            Method.PROCESS_CONTOUR:             (Method.END, ProcessContour(self._state)),

            Method.SELECT_METHOD:               (Method.END, SelectDetectContourMethod(self._state)),
            Method.CANNY:                       (Method.END, CannyMethod(self._state)),
            Method.ADAPTIVE:                    (Method.END, Adaptive(self._state))
        }  # переходы между состояниями

    def __call__(self, frame:np.ndarray)->np.ndarray:
        self._state.method = Method.DETECT_RECT_MARKERS

        self._state.current_frame = frame.copy()

        while self._state.method != Method.END:
            if self._state.method == Method.ERROR:
                return frame  # При ошибке возвращаем необработанный кадр
            next_method, method = self._transition[ self._state.method]
            self._state.method = next_method
            method()

        return self._state.current_frame