from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class Method(Enum):
    START = auto()  # вход в обработку
    END = auto()  # выход из обработки
    ERROR = auto()  # ошибка в процессе выполнения

    DETECT_RECT_MARKERS = auto()
    DETECT_LIGHT_MARKER = auto()
    CREATE_HOMOGRAPHY_TRANSFORM = auto()
    DRAW_PLANE = auto()
    FIND_CONTOUR = auto()

    SELECT_METHOD = auto()
    CANNY = auto()
    ADAPTIVE = auto()

@dataclass
class State:
    """Хранит переменные, которые используют функции обнаружения"""
    method:Method = Method.START

    centers = []
    src_points = []

    current_frame:Optional[np.ndarray] = None
    current_contour = None
    marker_data = None

