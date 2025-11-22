from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Environment:
    """Хранит переменные, которые используют функции обнаружения"""
    detector = None
    aruco_dict = None
    aruco_params = None

    centers = []
    src_points = []

    current_frame:Optional[np.ndarray] = None

