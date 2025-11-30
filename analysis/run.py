from logging import getLogger

from analysis.strategy.main_strategy import MainAnalysisStrategy
from logger_config import setup_logging
import cv2
import numpy as np

from scene3d.run3d import Run3D


class RunTime:
    """Замена сайта в окне"""
    obj:'RunTime' = None

    def __init__(self):
        self.obj = self
        setup_logging()
        self.logger = getLogger(type(self).__name__)

        self.analise_strategy = MainAnalysisStrategy()
        self.cap = cv2.VideoCapture(0)


    def __call__(self):
        has_viz = hasattr(cv2, 'viz')
        if has_viz:
            run3d = Run3D(self.analise_strategy.state)
            run3d.setup()
        while True:
            if has_viz:
                run3d.show()
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error('Failed to capture frame')

            result_frame = self.analise_strategy(frame)

            cv2.imshow('Original', result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break



if __name__ == '__main__':
    runtime = RunTime()
    runtime()