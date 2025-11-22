from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

from analysis.analysis_environment import Environment


class Function(ABC):
    """Функции, на которые разбивается алгоритм"""
    def __init__(self, environment:Environment):
        self.logger = getLogger(type(self).__name__)
        self.environment = environment

    @abstractmethod
    def __call__(self, *args, **kwargs)->Any: ...