from analysis.analysis_state import State, Method

import numpy as np
import cv2
import base64
from logging import getLogger

from analysis.functions.adaptive import Adaptive
from analysis.functions.canny import CannyMethod
from analysis.functions.create_homography_transform import CreateHomographyTransform
from analysis.functions.detect_light_marker import DetectLightMarker
from analysis.functions.detect_rect_markers import DetectRectMarkers
from analysis.functions.draw_plane import DrawPlane
from analysis.functions.find_contour import FindContour
from analysis.functions.select_detect_contour_method import SelectDetectContourMethod


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self._state = State()

        self.__logger = getLogger(type(self).__name__)

        self._transition = {
            Method.START:                       Method.DETECT_RECT_MARKERS,
            Method.DETECT_RECT_MARKERS:         Method.DETECT_LIGHT_MARKER,
            Method.DETECT_LIGHT_MARKER:         Method.CREATE_HOMOGRAPHY_TRANSFORM,
            Method.CREATE_HOMOGRAPHY_TRANSFORM: Method.DRAW_PLANE,
            Method.DRAW_PLANE:                  Method.FIND_CONTOUR,
            Method.FIND_CONTOUR:                Method.END,

            Method.CANNY:                       Method.END,
            Method.ADAPTIVE:                    Method.END
        }  # переходы между состояниями

        self.detect_rect_markers = DetectRectMarkers(self._state)
        self.detect_light_marker = DetectLightMarker(self._state)
        self.create_homography_transform = CreateHomographyTransform(self._state)
        self.draw_plane = DrawPlane(self._state)
        self.find_contour = FindContour(self._state)

        self.select_method = SelectDetectContourMethod(self._state)
        self.canny = CannyMethod(self._state)
        self.adaptive = Adaptive(self._state)

    def __call__(self, frame:np.ndarray)->np.ndarray:
        self._state.method = Method.START

        self._state.current_frame = frame

        while self._state.method != Method.END:
            match self._state.method:
                case Method.ERROR:
                    return frame  # при ошибке в процессе обработки возвращает необработанную картинку
                case Method.START:
                    self._state.method = self._transition[self._state.method]
                case Method.DETECT_RECT_MARKERS:
                    self._state.method = self._transition[self._state.method]
                    self.detect_rect_markers()
                case Method.DETECT_LIGHT_MARKER:
                    self._state.method = self._transition[self._state.method]
                    self.detect_light_marker()
                case Method.CREATE_HOMOGRAPHY_TRANSFORM:
                    self._state.method = self._transition[self._state.method]
                    self.create_homography_transform()
                case Method.DRAW_PLANE:
                    self._state.method = self._transition[self._state.method]
                    self.draw_plane()
                case Method.FIND_CONTOUR:
                    self._state.method = self._transition[self._state.method]
                    self.find_contour()
                case Method.SELECT_METHOD:
                    self.select_method()
                case Method.CANNY:
                    self._state.method = self._transition[self._state.method]
                    self.canny()
                case Method.ADAPTIVE:
                    self._state.method = self._transition[self._state.method]
                    self.adaptive()

        return self._state.current_frame