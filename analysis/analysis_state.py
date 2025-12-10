from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np


class Method(Enum):
    """Именованные константы для машины состояний"""
    EXIT = auto()  # выход из обработки
    ERROR = auto()  # ошибка в процессе выполнения

    DETECT_RECT_MARKERS = auto()
    CREATE_HOMOGRAPHY_TRANSFORM = auto()
    DRAW_PLANE = auto()

    FIND_CONTOUR = auto()
    PROCESS_CONTOUR = auto()

    HANDLE_SCANNING_DATA = auto()
    CREATE_MODEL = auto()

    DETECT_LIGHT_MARKER = auto()
    GET_SHADOW = auto()


@dataclass
class State:
    """Хранит переменные, которые используют функции обнаружения"""
    method:Method = Method.DETECT_RECT_MARKERS  # Текущий метод

    centers: List[np.ndarray] = field(default_factory=list)  # Центры маркеров
    src_points: List = field(default_factory=list)  # 2d координаты углов, являющиеся вершинами четырехугольника

    current_frame:Optional[np.ndarray] = None  # Текущий кадр
    contour: Optional[np.ndarray] = None  # Текущий контур в 2d
    marker_data: Optional[dict] = None  # Данные о маркерах

    plane_equation: Optional[Tuple[np.ndarray, float]] = None  # Уравнение плоскости четырехугольника
    current_contour_3d: List[List[np.ndarray]] = field(default_factory=list)  # Текущий контур в 3d

    scanning_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)  # Здесь хранятся контуры 3d и диагонали для создания 3d модели
    bottom_point = None  # 3d координата нижней точки контура

    dvecs:Optional[Tuple[np.ndarray, np.ndarray]] = None   # Пара векторов диагоналей в 3d
    start_vecs = None  # Пара начал диагоналей в 3d

    object3d:Optional[np.ndarray] = None  # Набор центров кубов
    cube_side:np.float32 = 0.0  # Длина стороны куба

    vertices = None  # вершины
    indices = None  # указатели вершин