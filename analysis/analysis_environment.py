from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Environment:
    """Хранит переменные, которые используют функции обнаружения"""
    detector = None
    current_frame:Optional[np.ndarray] = None