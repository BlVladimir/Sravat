from analysis.analysis_environment import State
from analysis.functions.function import Function

import cv2


class DetectLightMarker(Function):
    def __init__(self, environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        try:
            frame = self.env.current_frame
            corners, ids, rejected = self.env.detector_light_markers.detectMarkers(frame)

            if ids is None:
                return
            # Рисуем обнаруженные маркеры
            output_frame = frame.copy()
            cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)

            self.env.current_frame = output_frame
        except Exception as e:
            self.logger.error(f'Error detecting light marker: {e}')
            self.env.state = State.ERROR