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
        # 1. ПРОВЕРКИ
        if not self._validate_state():
            return

        # 2. ПОЛУЧЕНИЕ УГЛА ПАДЕНИЯ СВЕТА
        # Config.ANGLE_OF_INCIDENCE в градусах
        angle_incidence_rad = np.radians(Config.ANGLE_OF_INCIDENCE)

        # Угол поворота по Y из state (в радианах)
        y_axis_angle_rad = self._state.light_rotation

        # Суммарный угол
        total_light_angle_rad = angle_incidence_rad + y_axis_angle_rad

        # 3. ПОЛУЧЕНИЕ ДАННЫХ ИЗ STATE
        # vertices уже в системе координат диагоналей
        vertices = self._state.vertices
        indices = self._state.indices

        # dvecs содержит диагонали
        # dvecs[0] - главная диагональ (br_3d - tl_3d)
        # dvecs[1] - побочная диагональ (bl_3d - tr_3d)
        main_diag = self._state.dvecs[0]
        aux_diag = self._state.dvecs[1]

        # start_vecs содержит начала диагоналей в 3D (в системе камеры)
        # start_vecs[0] - начало главной диагонали (tl_3d)
        # start_vecs[1] - начало побочной диагонали (tr_3d)
        tl_3d = self._state.start_vecs[0]  # начало главной диагонали
        tr_3d = self._state.start_vecs[1]  # начало побочной диагонали

        # Вычисляем остальные углы
        br_3d = tl_3d + main_diag  # конец главной диагонали
        bl_3d = tr_3d + aux_diag  # конец побочной диагонали

        # 4. ПОСТРОЕНИЕ СИСТЕМЫ КООРДИНАТ ДИАГОНАЛЕЙ
        # Вспомним: vertices уже в этой системе, но нам нужно знать,
        # как преобразовывать точки из этой системы в пиксели

        # Построение ортонормированного базиса системы координат диагоналей:
        # X ось - вдоль главной диагонали
        x_axis = main_diag / np.linalg.norm(main_diag)

        # Z ось - нормаль к плоскости (векторное произведение диагоналей)
        z_axis = np.cross(main_diag, aux_diag)
        z_axis_norm = np.linalg.norm(z_axis)
        if z_axis_norm < 1e-10:
            # Диагонали коллинеарны
            z_axis = np.array([0, 0, 1.0], dtype=np.float32)
        else:
            z_axis = z_axis / z_axis_norm

        # Y ось - правая тройка
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Матрица преобразования ИЗ системы диагоналей В систему камеры
        # Каждая строка - ось системы диагоналей в координатах камеры
        R_diag_to_cam = np.array([x_axis, y_axis, z_axis])

        # 5. ПРОЕКЦИЯ ВЕРШИН НА ПЛОСКОСТЬ
        # Направление света в системе координат диагоналей
        # (предполагаем, что свет падает в плоскости XZ)
        light_direction_diag = np.array([
            np.cos(total_light_angle_rad),
            0.0,
            -np.sin(total_light_angle_rad)
        ], dtype=np.float32)
        light_direction_diag = light_direction_diag / np.linalg.norm(light_direction_diag)

        # Проецируем вершины на плоскость Z=0 в системе диагоналей
        shadow_points_2d = self._project_vertices_to_plane(vertices, light_direction_diag)

        # 6. ПОЛУЧЕНИЕ КООРДИНАТ УГЛОВ МАРКЕРОВ В СИСТЕМЕ ДИАГОНАЛЕЙ
        # Углы маркеров в системе камеры
        corners_3d_cam = np.array([tl_3d, tr_3d, br_3d, bl_3d], dtype=np.float32)

        # Преобразуем углы в систему диагоналей
        # Формула: P_diag = R_diag_to_cam @ (P_cam - tl_3d)
        corners_diag = []
        for corner in corners_3d_cam:
            corner_relative = corner - tl_3d
            corner_diag = R_diag_to_cam @ corner_relative
            corners_diag.append(corner_diag[:2])  # Берем только X,Y (Z должен быть ~0)

        corners_diag_2d = np.array(corners_diag, dtype=np.float32)

        # 7. РЕНДЕРИНГ ТЕНИ
        # Теперь у нас есть:
        # - shadow_points_2d: точки тени в системе диагоналей
        # - corners_diag_2d: углы маркеров в системе диагоналей
        # - src_points: углы маркеров на изображении (пиксели)

        # Рендерим тень
        shadow_image = self._render_shadow(
            shadow_points_2d, indices,
            corners_diag_2d, self._state.src_points
        )

        # 8. ОТОБРАЖЕНИЕ ТЕНИ НА КАДРЕ
        self._overlay_shadow(shadow_image)

        # 9. ЛОГИРОВАНИЕ
        self._log_debug_info(
            vertices, shadow_points_2d,
            angle_incidence_rad, y_axis_angle_rad,
            corners_diag_2d
        )

    def _validate_state(self):
        """Проверяет, что все необходимые данные есть в state"""
        required_attrs = [
            'vertices', 'indices', 'src_points',
            'dvecs', 'start_vecs', 'light_rotation'
        ]

        for attr in required_attrs:
            if not hasattr(self._state, attr) or getattr(self._state, attr) is None:
                self._logger.warning(f"В state отсутствует атрибут: {attr}")
                return False

        if len(self._state.vertices) == 0:
            self._logger.warning("Нет вершин для построения тени")
            return False

        if len(self._state.src_points) != 4:
            self._logger.warning(f"src_points должно содержать 4 точки, а содержит {len(self._state.src_points)}")
            return False

        if len(self._state.start_vecs) != 2:
            self._logger.warning(f"start_vecs должно содержать 2 точки, а содержит {len(self._state.start_vecs)}")
            return False

        return True

    def _project_vertices_to_plane(self, vertices, light_direction):
        """Проецирует вершины на плоскость Z=0 вдоль направления света"""
        shadow_points_2d = []

        for vertex in vertices:
            # Если свет параллелен плоскости
            if abs(light_direction[2]) < 1e-6:
                if abs(vertex[2]) < 1e-6:
                    shadow_points_2d.append([vertex[0], vertex[1]])
                else:
                    shadow_points_2d.append([vertex[0], vertex[1]])
                continue

            # Находим точку пересечения с плоскостью Z=0
            t = -vertex[2] / light_direction[2]
            shadow_point_3d = vertex + t * light_direction
            shadow_point_3d[2] = 0.0

            shadow_points_2d.append([shadow_point_3d[0], shadow_point_3d[1]])

        return np.array(shadow_points_2d, dtype=np.float32)

    def _render_shadow(self, shadow_points_2d, indices, corners_diag_2d, src_points):
        """Рендерит тень и преобразует в координаты изображения"""
        # src_points - углы маркеров на изображении
        src_points = np.array(src_points, dtype=np.float32)

        # 1. Находим гомографию из системы диагоналей в пиксели
        # corners_diag_2d -> src_points
        H, mask = cv2.findHomography(corners_diag_2d, src_points, cv2.RANSAC, 5.0)

        if H is None:
            self._logger.warning("Не удалось найти гомографию для преобразования тени")
            return None

        # 2. Преобразуем точки тени в пиксели
        shadow_points_pixel = cv2.perspectiveTransform(
            shadow_points_2d.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        # 3. Создаем изображение тени размером с кадр
        frame = self._state.current_frame
        h, w = frame.shape[:2]

        # Создаем маску для тени
        shadow_mask = np.zeros((h, w), dtype=np.uint8)

        # 4. Рисуем треугольники тени в маске
        for triangle_idx in indices:
            if len(triangle_idx) != 3:
                continue

            try:
                # Получаем точки треугольника в пикселях
                pts = shadow_points_pixel[triangle_idx].astype(np.int32)

                # Проверяем, что треугольник не вырожден
                area = abs(cv2.contourArea(pts))
                if area < 1.0:
                    continue

                # Рисуем заполненный треугольник в маске
                cv2.fillConvexPoly(shadow_mask, pts, 255)

            except Exception as e:
                self._logger.debug(f"Ошибка при рисовании треугольника: {e}")
                continue

        # 5. Применяем размытие для сглаживания краев
        kernel_size = 15  # Размер ядра размытия
        if kernel_size % 2 == 0:
            kernel_size += 1  # Делаем нечетным

        shadow_mask = cv2.GaussianBlur(shadow_mask, (kernel_size, kernel_size), 5)

        # 6. Создаем цветное изображение тени
        shadow_img = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_color = (80, 80, 80)  # Темно-серый цвет
        alpha_factor = 0.7  # Прозрачность

        # Заполняем цветом
        for c in range(3):
            shadow_img[:, :, c] = shadow_color[c]

        # Устанавливаем альфа-канал из маски
        shadow_img[:, :, 3] = (shadow_mask * alpha_factor).astype(np.uint8)

        return shadow_img

    def _overlay_shadow(self, shadow_image):
        """Накладывает тень на текущий кадр"""
        if shadow_image is None:
            return

        frame = self._state.current_frame

        # Проверяем размеры
        if shadow_image.shape[:2] != frame.shape[:2]:
            self._logger.warning(f"Размеры тени ({shadow_image.shape[:2]}) и кадра ({frame.shape[:2]}) не совпадают")
            return

        # Накладываем тень с помощью альфа-блендинга
        if shadow_image.shape[2] == 4:
            # Извлекаем альфа-канал
            alpha = shadow_image[:, :, 3] / 255.0

            # Накладываем тень
            for c in range(3):
                frame[:, :, c] = np.where(
                    alpha > 0,
                    # Основа - фон, затемненный в местах тени
                    frame[:, :, c] * (1.0 - alpha * 0.5) +
                    # Добавляем цвет тени
                    shadow_image[:, :, c] * alpha * 0.5,
                    frame[:, :, c]
                ).astype(np.uint8)

        self._state.current_frame = frame

    def _log_debug_info(self, vertices, shadow_points_2d, angle_incidence_rad,
                        y_axis_angle_rad, corners_diag_2d):
        """Логирует отладочную информацию"""
        self._logger.debug(f"Вершин: {len(vertices)}, точек тени: {len(shadow_points_2d)}")
        self._logger.debug(f"Угол падения: {np.degrees(angle_incidence_rad):.1f}°")
        self._logger.debug(f"Угол поворота: {np.degrees(y_axis_angle_rad):.1f}°")
        self._logger.debug(f"Суммарный угол: {np.degrees(angle_incidence_rad + y_axis_angle_rad):.1f}°")
        self._logger.debug(f"Координаты углов в системе диагоналей:")
        for i, corner in enumerate(corners_diag_2d):
            self._logger.debug(f"  Угол {i}: ({corner[0]:.3f}, {corner[1]:.3f})")