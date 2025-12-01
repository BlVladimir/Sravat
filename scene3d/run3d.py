from logging import debug

import cv2
import numpy as np

from analysis.analysis_state import State


class Run3D:
    def __init__(self, state:State):
        """Создает окно Viz с координатными осями. Для работы требует установки OpenCV с Viz"""
        self.state = state
        self.win = cv2.viz.Viz3d("3D Coordinate System")
        self.markers = set()

    def setup(self):
        # Создаем виджет координатных осей
        axes = cv2.viz.WCoordinateSystem(1.0)

        # Добавляем оси в окно
        self.win.showWidget("axes", axes)

    def show(self):
        for marker_name in self.markers:
            self.win.removeWidget(marker_name)
        self.markers.clear()

        # Если есть данные о расположении маркеров
        if self.state.marker_data is not None:
            for idx, data in enumerate(self.state.marker_data.values()):
                tvec = data['tvec']

                if tvec is None:
                    continue

                point = np.array(tvec).flatten()
                marker_name = f"marker_{idx}"
                sphere = cv2.viz.WSphere(point, radius=0.02, color=cv2.viz.Color.green())
                self.win.showWidget(marker_name, sphere)

                self.markers.add(marker_name)

        self.win.spinOnce(1, True)