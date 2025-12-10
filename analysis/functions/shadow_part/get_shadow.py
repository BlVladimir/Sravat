from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions

import numpy as np
import cv2


class GetShadow(Function):
    """
    Проецирует 3D модель на плоскость четырехугольника маркеров для получения тени
    """

    def __init__(self, state: State):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        angle_incidence_deg = np.radians(Config.ANGLE_OF_INCIDENCE)

        y_axis_angle_rad = self._state.light_rotation

        total_light_angle_rad = angle_incidence_deg + y_axis_angle_rad

        vertices = self._state.vertices
        indices = self._state.indices

        main_diag = self._state.dvecs[0]
        aux_diag = self._state.dvecs[1]

        marker_id_tl = self._state.start_vecs[0]
        origin_3d = self._state.marker_data[marker_id_tl]['tvec']

        x_axis = main_diag / np.linalg.norm(main_diag)

        z_axis = np.cross(main_diag, aux_diag)
        if np.linalg.norm(z_axis) < 1e-10:
            z_axis = np.array([0, 0, 1.0], dtype=np.float32)
        else:
            z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        basis_matrix = np.array([x_axis, y_axis, z_axis])

        light_direction_local = np.array([
            np.cos(total_light_angle_rad),
            0.0,
            -np.sin(total_light_angle_rad)
        ], dtype=np.float32)

        light_direction = basis_matrix.T @ light_direction_local
        light_direction = light_direction / np.linalg.norm(light_direction)

        shadow_points_3d = []

        for vertex in vertices:
            vertex_relative = vertex - origin_3d
            numerator = -np.dot(z_axis, vertex_relative)
            denominator = np.dot(z_axis, light_direction)

            if abs(denominator) < 1e-6:
                shadow_point = vertex
            else:
                t = numerator / denominator
                shadow_point = vertex + t * light_direction

            shadow_points_3d.append(shadow_point)

        shadow_points_3d = np.array(shadow_points_3d, dtype=np.float32)
        shadow_points_2d = []
        for point in shadow_points_3d:
            point_relative = point - origin_3d
            point_local = basis_matrix @ point_relative

            shadow_points_2d.append([point_local[0], point_local[1]])

        shadow_points_2d = np.array(shadow_points_2d, dtype=np.float32)
        shadow_image = self._render_shadow_to_image(shadow_points_2d, indices)

        self._overlay_shadow_on_frame(shadow_image)

    def _render_shadow_to_image(self, shadow_points_2d, indices, image_size=1024):
        """Рендер тени в изображение с сглаживанием"""
        min_coords = np.min(shadow_points_2d, axis=0)
        max_coords = np.max(shadow_points_2d, axis=0)

        margin = 0.15
        size = max_coords - min_coords
        min_coords -= size * margin
        max_coords += size * margin
        size = max_coords - min_coords

        scale = (image_size - 1) / np.max(size)

        shadow_img = np.zeros((image_size, image_size, 4), dtype=np.uint8)
        shadow_img[:, :, 3] = 0

        scaled_points = (shadow_points_2d - min_coords) * scale
        scaled_points = scaled_points.astype(np.float32)

        for triangle_idx in indices:
            pts = scaled_points[triangle_idx].copy()
            pts[:, 1] = image_size - 1 - pts[:, 1]

            triangle_mask = np.zeros((image_size, image_size), dtype=np.uint8)
            cv2.fillConvexPoly(triangle_mask, pts.astype(np.int32), 255)

            triangle_mask = cv2.GaussianBlur(triangle_mask, (5, 5), 1.5)

            shadow_img[:, :, 3] = np.maximum(shadow_img[:, :, 3], triangle_mask)

        shadow_img[:, :, 0] = 40
        shadow_img[:, :, 1] = 40
        shadow_img[:, :, 2] = 40

        shadow_img[:, :, 3] = (shadow_img[:, :, 3].astype(float) * 0.7).astype(np.uint8)

        return shadow_img

    def _overlay_shadow_on_frame(self, shadow_image):
        """Накладывает сглаженную тень на текущий кадр"""
        frame = self._state.current_frame
        src_points = np.array(self._state.src_points, dtype=np.float32)

        h, w = shadow_image.shape[:2]
        dst_points = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(dst_points, src_points)

        shadow_warped = cv2.warpPerspective(shadow_image, H, (frame.shape[1], frame.shape[0]),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        shadow_mask = shadow_warped[:, :, 3] / 255.0

        shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 2.0)

        shadow_mask = np.clip(shadow_mask, 0, 1)
        shadow_color = shadow_warped[:, :, :3]
        alpha = shadow_mask[:, :, np.newaxis]

        frame = frame.astype(float)
        shadow_color = shadow_color.astype(float)
        result = frame * (1 - alpha) + shadow_color * alpha

        frame = np.clip(result, 0, 255).astype(np.uint8)

        self._state.current_frame = frame