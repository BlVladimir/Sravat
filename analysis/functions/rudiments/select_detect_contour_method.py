from analysis.analysis_state import Method
from analysis.functions.function import Function, handle_exceptions

import cv2
import numpy as np


class SelectDetectContourMethod(Function):
    """Выбирает лучший способ выделить контур"""
    def __init__(self, state):
        super().__init__(state)

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        if len(self._state.centers) == 4 and self._state.src_points is not None:
            frame = self._state.current_frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self._state.src_points is not None and len(self._state.src_points) == 4:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                pts = np.int32(self._state.src_points)
                cv2.fillPoly(mask, [pts], 255)

                # Применяем маску к изображению
                gray = cv2.bitwise_and(gray, gray, mask=mask)

            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian > 100:
                self._state.method = Method.CANNY
                # self.logger.info('Using Canny edge detector')
            else:
                self._state.method = Method.ADAPTIVE
                # self.logger.info('Using adaptive thresholding')
        else:
            self._state.method = Method.END