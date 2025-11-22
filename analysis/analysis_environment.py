from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class State(Enum):
    START = auto()  # вход в обработку
    END = auto()  # выход из обработки
    ERROR = auto()  # ошибка в процессе выполнения

    DETECT_RECT_MARKERS = auto()
    DETECT_LIGHT_MARKER = auto()
    CREATE_HOMOGRAPHY_TRANSFORM = auto()
    DRAW_PLANE = auto()

@dataclass
class Environment:
    """Хранит переменные, которые используют функции обнаружения"""
    state:State = State.START

    detector_rect_markers = None
    aruco_rect_dict = None
    aruco_rect_params = None

    detector_light_markers = None
    aruco_light_dict = None
    aruco_light_params = None

    centers = []
    src_points = []

    current_frame:Optional[np.ndarray] = None

