from analysis.analysis_environment import Environment, State

import numpy as np
import cv2
import base64
from logging import getLogger

from analysis.functions.create_homography_transform import CreateHomographyTransform
from analysis.functions.detect_light_marker import DetectLightMarker
from analysis.functions.detect_rect_markers import DetectRectMarkers
from analysis.functions.draw_plane import DrawPlane


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self.env = Environment()

        self.env.aruco_rect_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.env.aruco_rect_params = cv2.aruco.DetectorParameters()
        self.env.detector_rect_markers = cv2.aruco.ArucoDetector(self.env.aruco_rect_dict, self.env.aruco_rect_params)

        self.env.aruco_light_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.env.aruco_light_params = cv2.aruco.DetectorParameters()
        self.env.detector_light_markers = cv2.aruco.ArucoDetector(self.env.aruco_light_dict, self.env.aruco_light_params)


        self.logger = getLogger(type(self).__name__)

        self._transition = {
            State.START:                       State.DETECT_RECT_MARKERS,
            State.DETECT_RECT_MARKERS:         State.DETECT_LIGHT_MARKER,
            State.DETECT_LIGHT_MARKER:         State.CREATE_HOMOGRAPHY_TRANSFORM,
            State.CREATE_HOMOGRAPHY_TRANSFORM: State.DRAW_PLANE,
            State.DRAW_PLANE:                  State.END
        }  # переходы между состояниями

        self.detect_rect_markers = DetectRectMarkers(self.env)
        self.detect_light_marker = DetectLightMarker(self.env)
        self.create_homography_transform = CreateHomographyTransform(self.env)
        self.draw_plane = DrawPlane(self.env)

    def __call__(self, base64_input:str)->str:
        self.env.state = State.START
        frame = self.to_cv2(base64_input)

        self.env.current_frame = frame

        while self.env.state != State.END:
            match self.env.state:
                case State.ERROR:
                    return base64_input  # при ошибке в процессе обработки возвращает необработанную картинку
                case State.START:
                    self.env.state = self._transition[self.env.state]
                case State.DETECT_RECT_MARKERS:
                    self.env.state = self._transition[self.env.state]
                    self.detect_rect_markers()
                case State.DETECT_LIGHT_MARKER:
                    self.env.state = self._transition[self.env.state]
                    self.detect_light_marker()
                case State.CREATE_HOMOGRAPHY_TRANSFORM:
                    self.env.state = self._transition[self.env.state]
                    self.create_homography_transform()
                case State.DRAW_PLANE:
                    self.env.state = self._transition[self.env.state]
                    self.draw_plane()

        result_base64 = self.to_base64(self.env.current_frame)

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