from analysis.analysis_state import State, Method
from analysis.functions.shadow_part.detect_light_marker import DetectLightMarker
from analysis.functions_group.functions_group import FunctionsGroup


class ShadowHandler(FunctionsGroup):
    def __init__(self, state: State):
        super().__init__(state)
        self._STARTED_METHOD = Method.DETECT_LIGHT_MARKER
        self._transition = {
            Method.DETECT_LIGHT_MARKER:    (Method.EXIT, DetectLightMarker(self._state))
        }

    def __bool__(self):
        return True