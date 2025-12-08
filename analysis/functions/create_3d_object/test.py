import unittest as ut
from itertools import repeat
from unittest.mock import Mock

import numpy as np
from scanning_optimized import scanning_optimized

from analysis.analysis_state import State
from analysis.functions.create_3d_object.handle_scanning_data import HandleScanningData


class Test(ut.TestCase):
    def setUp(self):
        mock_state = Mock(spec=State)
        self.handler = HandleScanningData(mock_state, 4)

    def test_calculate_parallelepiped(self):
        expected = np.array([[x/6, y/6, z/6]
                            for x in range(1, 6, 2)
                            for y in range(-1, 6, 2)
                            for z in range(1, 6, 2)])
        main_vec = np.array([3, 0, 0])
        auxiliary_vec = np.array([0, 4, 0])
        origin_main_pnt = np.array([0, 2, 0])
        origin_auxiliary_pnt = np.array([1, 1, 0])

        actual = self.handler._calculate_parallelepiped(main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt)

        np.testing.assert_array_almost_equal(actual, expected, decimal=6)

    def test_transform_to_local_coordinates(self):
        expected = np.array([[0, 0, 0],
                             [0, -1, 0],
                             [0, -1, -1],
                             [0, 0, -1]], dtype=np.float32)
        data = (np.array([2, 0, 0]), np.array([0, -2, 0]), np.array([1, 1, 1]), ..., np.array([[1, 1, 1], [1, 3, 1], [1, 3, 3], [1, 1, 3]]))
        actual, _ = HandleScanningData._transform_to_local_coordinates(data)
        np.testing.assert_array_almost_equal(actual, expected, decimal=6)

    def test1_process_contour(self):
        expected = list(repeat(0, 125))
        parallelepiped = np.array([[x, y, z]
                          for x in range(0, 5)
                          for y in range(-2, 3)
                          for z in range(0, 5)], dtype=np.float32)
        contours = [(np.array([[5, -10, -10],
                               [5, 10, -10],
                               [5, 10, 10],
                               [5, -10, 10]], dtype=np.float32),
                     np.array([[0, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 0]], dtype=np.float32))]

        actual = scanning_optimized.process_contours_optimized(parallelepiped, contours)
        np.testing.assert_array_almost_equal(actual, expected)

    def test2_process_contour(self):
        expected = np.array([1, 1, 1, 0, 1, 1, 1, 0])
        parallelepiped = np.array([[x, y, z]
                          for x in range(0, 2)
                          for y in range(0, 2)
                          for z in range(0, 2)], dtype=np.float32)
        contours = [(np.array([[1, 0.5, 1.5],
                               [1, 1.5, 1.5],
                               [1, 1.5, 0.5],
                               [1, 0.5, 0.5]], dtype=np.float32),
                     np.array([[0, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 0]], dtype=np.float32))]

        actual = scanning_optimized.process_contours_optimized(parallelepiped, contours)
        np.testing.assert_array_almost_equal(actual, expected)

if __name__ == '__main__':
    ut.main()