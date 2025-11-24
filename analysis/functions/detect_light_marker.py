from analysis.functions.function import Function, handle_exceptions

import cv2


class DetectLightMarker(Function):
    def __init__(self, state):
        super().__init__(state)
        self.aruco_light_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_light_params = cv2.aruco.DetectorParameters()
        self.detector_light_markers = cv2.aruco.ArucoDetector(self.aruco_light_dict, self.aruco_light_params)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        frame = self.state.current_frame
        corners, ids, rejected = self.detector_light_markers.detectMarkers(frame)

        if ids is None:
            return
        # Рисуем обнаруженные маркеры
        output_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)

        self.state.current_frame = output_frame