from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np


class Method(Enum):
    EXIT = auto()  # выход из обработки
    ERROR = auto()  # ошибка в процессе выполнения

    DETECT_RECT_MARKERS = auto()
    CREATE_HOMOGRAPHY_TRANSFORM = auto()
    DRAW_PLANE = auto()

    FIND_CONTOUR = auto()
    PROCESS_CONTOUR = auto()

    DETECT_LIGHT_MARKER = auto()


@dataclass
class State:
    """Хранит переменные, которые используют функции обнаружения"""
    method:Method = Method.DETECT_RECT_MARKERS

    centers: List[np.ndarray] = field(default_factory=list)
    src_points: List = field(default_factory=list)

    current_frame:Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    marker_data: Optional[dict] = None

    plane_equation: Optional[Tuple[np.ndarray, float]] = None
    current_contour_3d: List[List[np.ndarray]] = field(default_factory=list)

    scanning_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    bottom_point = None

    dvecs:Optional[Tuple[np.ndarray, np.ndarray]] = None
    start_vecs = None

    object3d:Optional[np.ndarray] = None