from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
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
    PROCESS_CONTOUR = auto()

    SELECT_METHOD = auto()
    CANNY = auto()
    ADAPTIVE = auto()

@dataclass
class State:
    """Хранит переменные, которые используют функции обнаружения"""
    method:Method = Method.DETECT_RECT_MARKERS

    centers: List[np.ndarray] = field(default_factory=list)
    src_points: List = field(default_factory=list)

    current_frame:Optional[np.ndarray] = None
    contour: Optional[List] = None
    marker_data: Optional[dict] = None

    plane_equation: Optional[Tuple[np.ndarray, float]] = None
    current_contour_3d: List[List[np.ndarray]] = field(default_factory=list)

