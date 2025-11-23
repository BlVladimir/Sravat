from analysis.analysis_environment import State
from analysis.functions.function import Function

import cv2
import numpy as np


class SelectDetectContourMethod(Function):
    """Выбирает лучший способ выделить контур"""
    def __init__(self, environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        try:
            if len(self.env.centers) == 4 and self.env.src_points is not None:
                frame = self.env.current_frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.env.src_points is not None and len(self.env.src_points) == 4:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    pts = np.int32(self.env.src_points)
                    cv2.fillPoly(mask, [pts], 255)

                    # Применяем маску к изображению
                    gray = cv2.bitwise_and(gray, gray, mask=mask)

                laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian > 100:
                    self.env.state = State.CANNY
                    # self.logger.info('Using Canny edge detector')
                else:
                    self.env.state = State.ADAPTIVE
                    # self.logger.info('Using adaptive thresholding')
            else:
                self.env.state = State.END
        except Exception as e:
            self.logger.error(f'Error selecting detect contour method: {e}')
            self.env.state = State.ERROR