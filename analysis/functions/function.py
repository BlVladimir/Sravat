import traceback
from abc import ABC, abstractmethod
from functools import wraps
from logging import getLogger
from typing import Any

from analysis.analysis_state import State, Method

def handle_exceptions(func):
    """Декоратор для обработки исключений в методах Function"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self._logger.error("Error in %s: %s", func.__name__, traceback.format_exc())
            # self._logger.error(e)
            self._state.method = Method.ERROR
    return wrapper

class Function(ABC):
    """Функции, на которые разбивается алгоритм"""
    def __init__(self, state:State):
        self._logger = getLogger(type(self).__name__)
        self._state = state

    @abstractmethod
    def __call__(self, *args, **kwargs)->Any: ...
