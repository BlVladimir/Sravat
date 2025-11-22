from analysis.analysis_environment import Environment, State

import numpy as np
import cv2
import base64
from logging import getLogger

from analysis.functions.create_homography_transform import CreateHomographyTransform
from analysis.functions.detect_markers import DetectMarkers
from analysis.functions.draw_plane import DrawPlane


class MainAnalysisStrategy:
    """Основная стратегия обработки"""
    def __init__(self):
        self.env = Environment()

        self.env.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.env.aruco_params = cv2.aruco.DetectorParameters()
        self.env.detector = cv2.aruco.ArucoDetector(self.env.aruco_dict, self.env.aruco_params)


        self.logger = getLogger(type(self).__name__)

        self._transition = {
            State.START:                       State.DETECT_MARKERS,
            State.ERROR:                       State.END,
            State.DETECT_MARKERS:              State.CREATE_HOMOGRAPHY_TRANSFORM,
            State.CREATE_HOMOGRAPHY_TRANSFORM: State.DRAW_PLANE,
            State.DRAW_PLANE:                  State.END
        }

        self.detect_markers = DetectMarkers(self.env)
        self.create_homography_transform = CreateHomographyTransform(self.env)
        self.draw_plane = DrawPlane(self.env)

        self.state = State.START

    def __call__(self, base64_input:str)->str:
        self.state = State.START
        frame = self.to_cv2(base64_input)

        self.env.current_frame = frame

        while self.state != State.END:
            match self.state:
                case State.ERROR:
                    return base64_input
                case State.START:
                    self.state = self._transition[self.state]
                case State.DETECT_MARKERS:
                    self.state = self._transition[self.state]
                    self.detect_markers()
                case State.CREATE_HOMOGRAPHY_TRANSFORM:
                    self.state = self._transition[self.state]
                    self.create_homography_transform()
                case State.DRAW_PLANE:
                    self.state = self._transition[self.state]
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