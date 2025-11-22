from analysis.analysis_environment import State
from analysis.functions.function import Function

import cv2


class SelectDetectContourMethod(Function):
    """Выбирает лучший способ выделить контур"""
    def __init__(self, environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        try:
            frame = self.env.current_frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian > 100:
                self.env.state = State.CANNY
            else:
                self.env.state = State.CANNY
        except Exception as e:
            self.logger.error(f'Error selecting detect contour method: {e}')
            self.env.state = State.ERROR