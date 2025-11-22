from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class State(Enum):
    START = auto()  # вход в обработку
    END = auto()  # выход из обработки
    ERROR = auto()  # ошибка в процессе выполнения

    DETECT_MARKERS = auto()
    CREATE_HOMOGRAPHY_TRANSFORM = auto()
    DRAW_PLANE = auto()

@dataclass
class Environment:
    """Хранит переменные, которые используют функции обнаружения"""
    state:State = State.START

    detector = None
    aruco_dict = None
    aruco_params = None

    centers = []
    src_points = []

    current_frame:Optional[np.ndarray] = None

