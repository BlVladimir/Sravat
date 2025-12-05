import unittest as ut

import numpy as np

from analysis.functions.create_3d_object.handle_scanning_data import HandleScanningData


class Test(ut.TestCase):
    def test_calculate_parallelepiped(self):
        actual = np.array([np.array([x/6, y/8, z/8])
                        for x in range(1, 6, 2)
                        for y in range(-5, 5, 2)
                        for z in range(1, 8, 2)])
        main_vec = np.array([3, 0, 0])
        auxiliary_vec = np.array([0, 4, 0])
        origin_main_pnt = np.array([0, 2, 0])
        origin_auxiliary_pnt = np.array([1, 0, 0])

        expected = HandleScanningData._calculate_parallelepiped(main_vec, auxiliary_vec, origin_main_pnt, origin_auxiliary_pnt)

        np.testing.assert_array_almost_equal(expected, actual, decimal=6)


if __name__ == '__main__':
    ut.main()