from unittest import case

from analysis.after_marker_manager.contour_handler import ContourHandler
from analysis.analysis_state import State, Method

import numpy as np
from logging import getLogger

from analysis.functions.markers_part.create_homography_transform import CreateHomographyTransform
from analysis.functions.shadow_part.detect_light_marker import DetectLightMarker
from analysis.functions.markers_part.detect_rect_markers import DetectRectMarkers
from analysis.functions.markers_part.draw_plane import DrawPlane


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self._state = State()

        self._logger = getLogger(type(self).__name__)

        self._transition = {
            Method.DETECT_RECT_MARKERS:         (Method.CREATE_HOMOGRAPHY_TRANSFORM, DetectRectMarkers(self._state)),
            Method.CREATE_HOMOGRAPHY_TRANSFORM: (Method.DRAW_PLANE, CreateHomographyTransform(self._state)),
            Method.DETECT_LIGHT_MARKER:         (Method.DRAW_PLANE, DetectLightMarker(self._state)),
            Method.DRAW_PLANE:                  (Method.END_MARKER_PART, DrawPlane(self._state))
        }  # переходы между состояниями

        self._contour_handler = ContourHandler(self._state)

    def __call__(self, frame:np.ndarray)->np.ndarray:
        self._state.method = Method.DETECT_RECT_MARKERS

        self._state.current_frame = frame.copy()

        while self._state.method != Method.END:
            match self._state.method:
                case Method.ERROR:
                    return frame  # При ошибке возвращаем необработанный кадр
                case Method.DETECT_RECT_MARKERS | Method.CREATE_HOMOGRAPHY_TRANSFORM | Method.DRAW_PLANE:
                    next_method, method = self._transition[self._state.method]
                    self._state.method = next_method
                    method()
                case _:
                    self._contour_handler.process_frame()

        return self._state.current_frame