from analysis.analysis_environment import State
from analysis.functions.function import Function, handle_exceptions

import cv2


class DetectLightMarker(Function):
    def __init__(self, environment):
        super().__init__(environment)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self.env.current_frame
        corners, ids, rejected = self.env.detector_light_markers.detectMarkers(frame)

        if ids is None:
            return
        # Рисуем обнаруженные маркеры
        output_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)

        self.env.current_frame = output_frame