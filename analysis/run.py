from logging import getLogger
from unittest import case

from analysis.facade_analysis import FacadeAnalysis
from logger_config import setup_logging
import cv2

from sandbox.scene3d.run3d import Run3D


class RunTime:
    """Основной класс"""
    obj:'RunTime' = None  # одиночка

    def __init__(self):
        self.obj = self
        setup_logging()
        self.logger = getLogger(type(self).__name__)

        self.facade = FacadeAnalysis()
        self.cap = cv2.VideoCapture(0)


    def __call__(self):
        # has_viz = hasattr(cv2, 'viz')
        # # has_viz = False
        # if has_viz:
        #     run3d = Run3D(self.facade._main_strategy._state)
        #     run3d.setup()
        while True:
            # if has_viz:
            #     run3d.show()
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error('Failed to capture frame')
                continue

            result_frame = self.facade.analyze_frame(frame)

            cv2.imshow('Original', result_frame)

            key = cv2.waitKey(3) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.facade.reset()
            elif key == ord('c'):
                self.facade.recalibrate()

if __name__ == '__main__':
    runtime = RunTime()
    runtime()