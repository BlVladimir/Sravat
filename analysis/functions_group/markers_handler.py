from analysis.analysis_state import State, Method
from analysis.functions.markers_part.create_homography_transform import CreateHomographyTransform
from analysis.functions.markers_part.detect_rect_markers import DetectRectMarkers
from analysis.functions.markers_part.draw_plane import DrawPlane
from analysis.functions.shadow_part.detect_light_marker import DetectLightMarker
from analysis.functions_group.functions_group import FunctionsGroup


class MarkersHandler(FunctionsGroup):
    """Обработка маркеров плоскости"""
    def __init__(self, state: State):
        super().__init__(state)
        self._STARTED_METHOD = Method.DETECT_RECT_MARKERS
        self._transition = {
            Method.DETECT_RECT_MARKERS:         (Method.CREATE_HOMOGRAPHY_TRANSFORM, DetectRectMarkers(self._state)),
            Method.CREATE_HOMOGRAPHY_TRANSFORM: (Method.DRAW_PLANE, CreateHomographyTransform(self._state)),
            Method.DETECT_LIGHT_MARKER:         (Method.DRAW_PLANE, DetectLightMarker(self._state)),
            Method.DRAW_PLANE:                  (Method.EXIT, DrawPlane(self._state))
        }