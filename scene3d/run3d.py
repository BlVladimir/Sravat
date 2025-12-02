from itertools import islice

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

        self.markers = set(f"marker_{idx}" for idx in range(4))
        for marker_name in self.markers:
            sphere = cv2.viz.WSphere(np.zeros(3, dtype=np.float32), radius=0.02, color=cv2.viz.Color.green())
            self.win.showWidget(marker_name, sphere)

    def show(self):
        # Если есть данные о расположении маркеров
        if self.state.marker_data is not None:
            for idx, data  in enumerate(islice(self.state.marker_data.values(), 4)):
                tvec = data['tvec']

                pose = cv2.viz.Affine3d(np.zeros(3, dtype=np.float32), tvec)

                self.win.setWidgetPose(f"marker_{idx}", pose)

        self.win.spinOnce(1, True)