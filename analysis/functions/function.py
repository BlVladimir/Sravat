from abc import ABC, abstractmethod
from functools import wraps
from logging import getLogger
from typing import Any

from analysis.analysis_environment import Environment, State

def handle_exceptions(func):
    """Декоратор для обработки исключений в методах Function"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f'Error in {type(self).__name__}: {e}')
            self.env.state = State.ERROR
    return wrapper

class Function(ABC):
    """Функции, на которые разбивается алгоритм"""
    def __init__(self, environment:Environment):
        self.logger = getLogger(type(self).__name__)
        self.env = environment

    @abstractmethod
    def __call__(self, *args, **kwargs)->Any: ...
