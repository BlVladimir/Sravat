from analysis.analysis_config import Config
from analysis.analysis_state import State, Method
from analysis.functions.create_3d_object.creating_model import CreatingModel
from analysis.functions.create_3d_object.handle_scanning_data import HandleScanningData
from analysis.functions_group.functions_group import FunctionsGroup


class ProcessData(FunctionsGroup):
    """Создание 3D модели из данных о контуре"""
    def __init__(self, state: State):
        super().__init__(state)
        self._STARTED_METHOD = Method.HANDLE_SCANNING_DATA
        self._transition = {
            Method.HANDLE_SCANNING_DATA: (Method.CREATE_MODEL, HandleScanningData(self._state, Config.EDGE)),
            Method.CREATE_MODEL:         (Method.EXIT, CreatingModel(self._state)),
        }