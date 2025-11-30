import cv2
import numpy as np

from analysis.analysis_state import State


class Run3D:
    def __init__(self, state:State):
        """Создает окно Viz с координатными осями. Для работы требует установки OpenCV с Viz"""
        self.state = state
        self.win = cv2.viz.Viz3d("3D Coordinate System")

    def setup(self):
        # Создаем виджет координатных осей
        axes = cv2.viz.WCoordinateSystem(1.0)

        # Добавляем оси в окно
        self.win.showWidget("axes", axes)

    def show(self):
        self.win.spinOnce(1, True)