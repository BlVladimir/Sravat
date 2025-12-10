import numpy as np
import cv2
from scanning_optimized import scanning_optimized

from analysis.analysis_state import State
from analysis.functions.function import Function, handle_exceptions


class CreatingModel(Function):
    """Создает модель"""
    def __init__(self, state:State):
        super().__init__(state)
        self._filename:str = 'model.obj'  # название файла, в котором будет лежать модель в формате .obj

    @handle_exceptions
    def __call__(self, *args, **kwargs):
        centers, cube_side = self._state.object3d, self._state.cube_side
        vertices, indices, normals = scanning_optimized.build_voxel_mesh_with_normals(centers, cube_side)
        self._state.vertices = vertices
        self._state.indices = indices
        self._write_obj(vertices, indices, normals)


    def reset(self):
        self._state.vertices = None
        self._state.indices = None

    def _write_obj(self, vertices:np.ndarray, indices:np.ndarray, normals:np.ndarray):
        """Записывает меш в файл формата OBJ"""
        with open(self._filename, 'w', encoding='utf-8') as f:
            f.write('# Generated from voxel mesh\n')
            f.write(f'# Vertices: {len(vertices)}, Faces: {len(indices)}\n\n')

            for v in vertices:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

            if normals is not None:
                f.write('\n')
                for n in normals:
                    f.write(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')

            f.write('\n')
            if normals is not None:
                for face in indices:
                    v1, v2, v3 = face + 1
                    f.write(f'f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n')
            else:
                for face in indices:
                    v1, v2, v3 = face + 1
                    f.write(f'f {v1} {v2} {v3}\n')

        self._logger.info(f'OBJ файл сохранен: {self._filename}')