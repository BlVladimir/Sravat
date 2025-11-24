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
from analysis.functions.select_detect_contour_method import SelectDetectContourMethod


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self.state = State()

        self.logger = getLogger(type(self).__name__)

        self._transition = {
            Method.START:                       Method.DETECT_RECT_MARKERS,
            Method.DETECT_RECT_MARKERS:         Method.DETECT_LIGHT_MARKER,
            Method.DETECT_LIGHT_MARKER:         Method.CREATE_HOMOGRAPHY_TRANSFORM,
            Method.CREATE_HOMOGRAPHY_TRANSFORM: Method.DRAW_PLANE,
            Method.DRAW_PLANE:                  Method.SELECT_METHOD,

            Method.CANNY:                       Method.END,
            Method.ADAPTIVE:                    Method.END
        }  # переходы между состояниями

        self.detect_rect_markers = DetectRectMarkers(self.state)
        self.detect_light_marker = DetectLightMarker(self.state)
        self.create_homography_transform = CreateHomographyTransform(self.state)
        self.draw_plane = DrawPlane(self.state)

        self.select_method = SelectDetectContourMethod(self.state)
        self.canny = CannyMethod(self.state)
        self.adaptive = Adaptive(self.state)

    def __call__(self, base64_input:str)->str:
        self.state.method = Method.START
        frame = self.to_cv2(base64_input)

        self.state.current_frame = frame

        while self.state.method != Method.END:
            match self.state.method:
                case Method.ERROR:
                    return base64_input  # при ошибке в процессе обработки возвращает необработанную картинку
                case Method.START:
                    self.state.method = self._transition[self.state.method]
                case Method.DETECT_RECT_MARKERS:
                    self.state.method = self._transition[self.state.method]
                    self.detect_rect_markers()
                case Method.DETECT_LIGHT_MARKER:
                    self.state.method = self._transition[self.state.method]
                    self.detect_light_marker()
                case Method.CREATE_HOMOGRAPHY_TRANSFORM:
                    self.state.method = self._transition[self.state.method]
                    self.create_homography_transform()
                case Method.DRAW_PLANE:
                    self.state.method = self._transition[self.state.method]
                    self.draw_plane()
                case Method.SELECT_METHOD:
                    self.select_method()
                case Method.CANNY:
                    self.state.method = self._transition[self.state.method]
                    self.canny()
                case Method.ADAPTIVE:
                    self.state.method = self._transition[self.state.method]
                    self.adaptive()

        result_base64 = self.to_base64(self.state.current_frame)

        return result_base64

    def to_cv2(self, base64_string: str):
        """Конвертирует Base64 строку в изображение OpenCV"""
        try:
            img_data = base64.b64decode(base64_string)

            np_array = np.frombuffer(img_data, np.uint8)

            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError('Failed to decode image from Base64')

            return image
        except Exception as e:
            self.logger.error(f'Error converting Base64 to OpenCV: {e}')
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def to_base64(self, image: np.ndarray):
        """Конвертирует изображение OpenCV в Base64 строку"""
        try:
            success, encoded_image = cv2.imencode('.jpg', image)

            if not success:
                raise ValueError('Failed to encode image to JPEG')

            base64_string = base64.b64encode(encoded_image).decode('utf-8')

            return base64_string
        except Exception as e:
            self.logger.error(f'Error converting OpenCV to Base64: {e}')
            return ''