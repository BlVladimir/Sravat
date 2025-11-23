from analysis.analysis_config import Config
from analysis.analysis_environment import Environment, State
from analysis.functions.function import Function

import cv2
import numpy as np


class DrawPlane(Function):
    def __init__(self, environment: Environment):
        super().__init__(environment)

    def __call__(self, *args, **kwargs):
        """Рисует плоскость и маркеры на кадре"""
        try:
            output_frame = self.env.current_frame
            src_points = self.env.src_points

            if len(self.env.centers) == 4 and src_points is not None:
                pts = src_points.astype(int)

                # Создаём маску четырёхугольника
                mask = np.zeros(output_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                # Инвертируем маску (всё вне четырёхугольника)
                mask_inverted = cv2.bitwise_not(mask)

                # Заливаем всё вне четырёхугольника белым
                output_frame[mask_inverted == 255] = [255, 255, 255]

                # Рисуем контур плоскости
                cv2.polylines(output_frame, [pts], True, Config.colors['contour'], 3)

                # Рисуем заливку плоскости с прозрачностью
                overlay = output_frame.copy()
                cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0, output_frame)

                # Рисуем угловые точки
                for i, (point, color) in enumerate(zip(pts, Config.colors['corners'])):
                    cv2.circle(output_frame, tuple(point), 3, color, -1)

            self.env.current_frame = output_frame
        except Exception as e:
            self.logger.error(f'Error drawing plane: {e}')
            self.env.state = State.ERROR