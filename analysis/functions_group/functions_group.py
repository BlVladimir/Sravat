from abc import ABC
from logging import getLogger
from typing import Tuple, Dict

from analysis.analysis_state import State, Method
from analysis.functions.function import Function


class FunctionsGroup(ABC):
    """Группа функций, у которых одна зона ответственности"""
    _transition: Dict[Method, Tuple[Method, Function]]  # cur_method:(next_method, function)
    _STARTED_METHOD: Method

    def __init__(self, state:State):
        self._logger = getLogger(type(self).__name__)
        self._state = state

    def __call__(self, *args, **kwargs):
        self._state.method = self._STARTED_METHOD
        while self._state.method != Method.EXIT and self._state.method != Method.ERROR:
            next_method, function = self._transition[self._state.method]
            self._state.method = next_method
            function()