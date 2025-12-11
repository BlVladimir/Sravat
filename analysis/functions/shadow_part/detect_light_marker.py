from analysis.analysis_config import Config
from analysis.analysis_state import Method
from analysis.functions.function import Function, handle_exceptions

import cv2
import numpy as np


class DetectLightMarker(Function):
    """Ищет маркер света"""
    def __init__(self, state):
        super().__init__(state)
        self.aruco_light_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_light_params = cv2.aruco.DetectorParameters()
        self.detector_light_markers = cv2.aruco.ArucoDetector(self.aruco_light_dict, self.aruco_light_params)

        self.marker_points_3d = np.array([
            [-Config.MARKER_LIGHT_SIZE/2, Config.MARKER_LIGHT_SIZE/2, 0],
            [Config.MARKER_LIGHT_SIZE/2, Config.MARKER_LIGHT_SIZE/2, 0],
            [Config.MARKER_LIGHT_SIZE/2, -Config.MARKER_LIGHT_SIZE/2, 0],
            [-Config.MARKER_LIGHT_SIZE/2, -Config.MARKER_LIGHT_SIZE/2, 0]
        ], dtype=np.float32)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        self._logger.info('Try to detect light marker...')
        frame = self._state.current_frame
        corners, ids, rejected = self.detector_light_markers.detectMarkers(frame)

        if ids is None or len(ids) != 1:
            self._state.method = Method.ERROR
            self._logger.error('Marker detection failed')
            return

        output_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)

        self._state.current_frame = output_frame

        marker_corners = corners[0]

        successful, rvec, tvec = cv2.solvePnP(
            self.marker_points_3d,
            marker_corners,
            Config.camera_matrix,
            Config.dist_coeffs
        )

        if successful:
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)

            y_axis_angle = euler_angles[1]

            self._state.light_rotation = y_axis_angle



    def rotation_matrix_to_euler_angles(self, R):
        """Преобразует матрицу вращения в углы Эйлера (XYZ) в радианах"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])