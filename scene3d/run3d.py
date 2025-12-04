from itertools import islice
from logging import getLogger

import cv2
import numpy as np

from analysis.analysis_state import State


class Run3D:
    def __init__(self, state:State):
        """Создает окно Viz с координатными осями. Для работы требует установки OpenCV с Viz"""
        self.state = state
        self.win = cv2.viz.Viz3d("3D Coordinate System")

        self.markers = set()
        self.contour_widgets = set()

        self._logger = getLogger(type(self).__name__)

    def setup(self):
        # Создаем виджет координатных осей
        axes = cv2.viz.WCoordinateSystem(1.0)

        # Добавляем оси в окно
        self.win.showWidget("axes", axes)

        self.markers = set(f"marker_{idx}" for idx in range(4))
        for marker_name in self.markers:
            sphere = cv2.viz.WSphere(np.zeros(3, dtype=np.float32), radius=0.02, color=cv2.viz.Color.green())
            self.win.showWidget(marker_name, sphere)

        sphere = cv2.viz.WSphere(np.zeros(3, dtype=np.float32), radius=0.02, color=cv2.viz.Color.green())
        self.win.showWidget('bottom_point', sphere)

    def show(self):
        # Если есть данные о расположении маркеров
        if self.state.marker_data is not None:
            for idx, data  in enumerate(islice(self.state.marker_data.values(), 4)):
                tvec = data['tvec']

                pose = cv2.viz.Affine3d(np.zeros(3, dtype=np.float32), tvec)

                self.win.setWidgetPose(f"marker_{idx}", pose)

        if self.state.bottom_point is not None:
            pose = cv2.viz.Affine3d(np.zeros(3, dtype=np.float32), self.state.bottom_point)
            self.win.setWidgetPose('bottom_point', pose)

        self.draw_contour()

        self.win.spinOnce(1, True)

    def draw_contour(self):
        """Рисует 3D контуры в окне Viz"""
        # Удаляем старые виджеты контуров
        for widget_name in self.contour_widgets:
            self.win.removeWidget(widget_name)
        self.contour_widgets.clear()


        # Если есть данные о 3D контурах
        if self.state.current_contour_3d:
            for contour_idx, contour_3d in enumerate(self.state.current_contour_3d):
                if len(contour_3d) < 2:
                    continue  # Пропускаем контуры с недостаточным количеством точек

                # Рисуем линии между точками контура
                for i in range(len(contour_3d)):
                    p1 = np.array(contour_3d[i], dtype=np.float32)
                    p2 = np.array(contour_3d[(i + 1) % len(contour_3d)], dtype=np.float32)

                    # Создаем линию между двумя точками
                    line_widget = cv2.viz.WLine(p1, p2, color=cv2.viz.Color.red())
                    widget_name = f"contour_{contour_idx}_line_{i}"

                    self.win.showWidget(widget_name, line_widget)
                    self.contour_widgets.add(widget_name)

                # Опционально: рисуем точки контура как маленькие сферы
                for point_idx, point in enumerate(contour_3d):
                    point_np = np.array(point, dtype=np.float32)
                    sphere = cv2.viz.WSphere(point_np, radius=0.005, color=cv2.viz.Color.blue())
                    widget_name = f"contour_{contour_idx}_point_{point_idx}"

                    self.win.showWidget(widget_name, sphere)
                    self.contour_widgets.add(widget_name)
