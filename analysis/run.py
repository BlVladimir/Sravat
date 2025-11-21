import base64
from logging import getLogger

from analysis.analysis_strategy import MainAnalysisStrategy
from logger_config import setup_logging
import cv2
import numpy as np


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
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error('Failed to capture frame')

            base64_data = self.frame_to_base64(frame)

            if base64_data:
                result_base64 = self.analise_strategy(base64_data)

                if result_base64:
                    processed_data = base64.b64decode(result_base64)
                    processed_array = np.frombuffer(processed_data, np.uint8)
                    processed_frame = cv2.imdecode(processed_array, cv2.IMREAD_COLOR)
                    cv2.imshow('Processed', processed_frame)
                else:
                    cv2.imshow('Original', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    def frame_to_base64(self, frame):
        """Конвертирует OpenCV frame в Base64 строку"""
        try:
            success, encoded_image = cv2.imencode('.jpg', frame)
            if success:
                base64_string = base64.b64encode(encoded_image).decode('utf-8')
                return base64_string
            else:
                return ''
        except Exception as e:
            self.logger.error(f'Error converting frame to Base64: {e}')
            return ''



if __name__ == '__main__':
    runtime = RunTime()
    runtime()