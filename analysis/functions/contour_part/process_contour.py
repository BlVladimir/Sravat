from analysis.analysis_config import Config
from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions

import numpy as np


class ProcessContour(Function):
    """Переводит контур в 3D координаты"""
    def __init__(self, state:State):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        self._state.current_contour_3d.clear()

        # if self._state.plane_equation is not None and Config.camera_matrix is not None:
        #     contour_3d = []
        #     for point in self._state.contour.reshape(-1, 2):
        #         point_3d = self.project_2d_to_3d(
        #             tuple(point),
        #             Config.camera_matrix,
        #             self._state.plane_equation
        #         )
        #         if point_3d is not None:
        #             contour_3d.append(point_3d)
        #     self._state.current_contour_3d.append(contour_3d)

        contour = self._state.contour
        points = contour.reshape(-1, 2)
        max_y_idx = np.argmax(points[:, 1])

        bottom_point = points[max_y_idx]

        bottom_point_3d = self.project_bottom_point_to_3d(bottom_point)
        self._state.scanning_data.append((self._state.dvecs[0], self._state.dvecs[1], self._state.start_vecs[0], self._state.start_vecs[1], bottom_point_3d))

        self._state.bottom_point = bottom_point_3d

        if bottom_point_3d is not None and Config.camera_matrix is not None:
            plane_normal = np.array([0.0, 0.0, -1.0])
            plane_d = -np.dot(plane_normal, bottom_point_3d)
            plane_equation = (plane_normal, plane_d)

            contour_3d = []
            for point in points:
                point_3d = self.project_2d_to_3d(
                    tuple(point),
                    Config.camera_matrix,
                    plane_equation
                )
                if point_3d is not None:
                    contour_3d.append(point_3d)
            self._state.current_contour_3d.append(contour_3d)

    def project_bottom_point_to_3d(self, bottom_point):
        """
        Определяет 3D координату нижней точки на основе её относительного положения в 2D
        """

        tl, tr, br, bl = self._state.src_points

        tl = np.array(tl, dtype=np.float32)
        tr = np.array(tr, dtype=np.float32)
        br = np.array(br, dtype=np.float32)
        bl = np.array(bl, dtype=np.float32)

        u, v = 0.5, 0.5
        for _ in range(10):
            top_interp = (1 - u) * tl + u * tr

            bottom_interp = (1 - u) * bl + u * br

            point_estimate = (1 - v) * top_interp + v * bottom_interp

            du_vec = (1 - v) * (tr - tl) + v * (br - bl)
            dv_vec = bottom_interp - top_interp

            diff = bottom_point - point_estimate

            jacobian = np.column_stack([du_vec, dv_vec])
            try:
                delta = np.linalg.lstsq(jacobian, diff, rcond=None)[0]
                u += delta[0]
                v += delta[1]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)

                if np.linalg.norm(diff) < 0.1:
                    break
            except:
                break

        corners_3d = []
        sorted_ids = sorted(self._state.marker_data.keys())[:4]

        for marker_id in sorted_ids:
            tvec = self._state.marker_data[marker_id]['tvec']
            corners_3d.append(tvec)
        tl_3d, tr_3d, br_3d, bl_3d = corners_3d

        top_interp_3d = (1 - u) * tl_3d + u * tr_3d
        bottom_interp_3d = (1 - u) * bl_3d + u * br_3d
        point_3d = (1 - v) * top_interp_3d + v * bottom_interp_3d

        n, d = self._state.plane_equation

        distance = np.dot(n, point_3d) + d

        point_3d_corrected = point_3d - distance * n

        return point_3d_corrected

    @staticmethod
    def project_2d_to_3d(point_2d, camera_matrix, plane_equation):
        """Проецирует 2D точку изображения в 3D на заданной плоскости"""
        n, d = plane_equation
        inv_K = np.linalg.inv(camera_matrix)
        point_2d_hom = np.array([point_2d[0], point_2d[1], 1.0])
        ray_dir = inv_K.dot(point_2d_hom)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        denominator = np.dot(n, ray_dir)
        if abs(denominator) < 1e-6:
            return None

        t = -d / denominator
        point_3d = t * ray_dir
        return point_3d

