import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors import DataFrame


class TestDataFrameInit(unittest.TestCase):
    def test_class_init(self):
        phi1 = DataFrame(['x1', 'x2', 'x3'], list(range(12)))
        self.assertListEqual(phi1.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi1.values, np.arange(12).reshape(-1, 3))

        phi2 = DataFrame(['x1', 'x2', 5, (1, 2)],
                         np.array([[0,  1,  2,  3],
                                   [4,  5,  6,  7],
                                   [8,  9, 10, 11],
                                   [12, 13, 14, 15],
                                   [16, 17, 18, 19]]))
        self.assertListEqual(phi2.variables, ['x1', 'x2', 5, (1, 2)])
        np_test.assert_array_equal(phi2.values, np.arange(20).reshape(-1, 4))

    def test_class_init_error(self):
        self.assertRaises(TypeError, DataFrame, 'x1', np.arange(12))
        self.assertRaises(TypeError, DataFrame, 2, np.arange(12))
        self.assertRaises(TypeError, DataFrame, ['x1', 'x2', 'x3'], ['a', 'b', 'c'])
        self.assertRaises(TypeError, DataFrame, ['x1', 'x2', 'x3'], 'A')
        self.assertRaises(ValueError, DataFrame, ['x1', 'x2', 'x3'], np.arange(10))
        self.assertRaises(ValueError, DataFrame, ['x1', 2], np.arange(13))


class TestDataFrameMethods1(unittest.TestCase):
    def setUp(self):
        self.phi1 = DataFrame(['x1', 'x2', 'x3'], list(range(12)))
        self.phi2 = DataFrame(['x1', 'x2', 5, (1, 2)],
                              np.array([[0,  1,  2,  3],
                                       [4,  5,  6,  7],
                                       [8,  9, 10, 11],
                                       [12, 13, 14, 15],
                                       [16, 17, 18, 19]]))

    def test_get_variables(self):
        self.assertListEqual(self.phi1.get_variables(), ['x1', 'x2', 'x3'])
        self.assertListEqual(self.phi2.get_variables(), ['x1', 'x2', 5, (1, 2)])

    def test_get_values(self):
        np_test.assert_array_equal(self.phi1.get_values(), np.arange(12).reshape(-1, 3))
        np_test.assert_array_equal(self.phi2.get_values(), np.arange(20).reshape(-1, 4))

    def test_get_num_of_samples(self):
        self.assertEqual(self.phi1.get_num_of_samples(), 4)
        self.assertEqual(self.phi2.get_num_of_samples(), 5)

    def test_add_samples(self):
        self.phi1.add_samples(list(range(12, 18)))
        np_test.assert_array_equal(self.phi1.values, np.arange(18).reshape(-1, 3))

        self.phi1.add_samples(list(range(18, 24)))
        np_test.assert_array_equal(self.phi1.values, np.arange(24).reshape(-1, 3))

        self.phi2.add_samples(np.array([[20, 21, 22, 23],
                                        [24, 25, 26, 27],
                                        [28, 29, 30, 31]]))
        np_test.assert_array_equal(self.phi2.values, np.arange(32).reshape(-1, 4))

        self.phi2.add_samples(list(range(32, 40)))
        np_test.assert_array_equal(self.phi2.values, np.arange(40).reshape(-1, 4))

    def test_add_samples_error(self):
        self.assertRaises(TypeError, self.phi1.add_samples, ['a', 'b', 'c'])
        self.assertRaises(TypeError, self.phi2.add_samples, ['a', 'b', 'c'])
        self.assertRaises(TypeError, self.phi1.add_samples, 'A')
        self.assertRaises(TypeError, self.phi2.add_samples, 'A')
        self.assertRaises(ValueError, self.phi1.add_samples, np.arange(13))
        self.assertRaises(ValueError, self.phi2.add_samples, np.arange(13))

    def tearDown(self):
        del self.phi1
        del self.phi2


class TestDataFrameMethods2(unittest.TestCase):
    def setUp(self):
        self.phi1 = DataFrame(['x1', 'x2', 'x3'], list(range(12)))

        self.phi2 = DataFrame(['x1', 'x2', 5, (1, 2)],
                              np.array([[0,  1,  2,  3],
                                       [4,  5,  6,  7],
                                       [8,  9, 10, 11],
                                       [12, 13, 14, 15],
                                       [16, 17, 18, 19]]))

        self.phi3 = DataFrame(['x2', 5, (1, 2), 'x1'],
                              np.array([[1,  2,  3,  0],
                                       [5,  6,  7,  4],
                                       [9, 10, 11,  8],
                                       [13, 14, 15, 12],
                                       [17, 18, 19, 16]]))

        self.phi4 = DataFrame(['x1', 'x2', 5, (1, 2)],
                              np.array([[4,  5,  6,  7],
                                       [8,  9, 10, 11],
                                       [0,  1,  2,  3],
                                       [16, 17, 18, 19],
                                       [12, 13, 14, 15]]))

        self.phi5 = DataFrame(['x2', 5, (1, 2), 'x1'],
                              np.array([[17, 18, 19, 16],
                                       [1,  2,  3,  0],
                                       [9, 10, 11,  8],
                                       [5,  6,  7,  4],
                                       [13, 14, 15, 12]]))

    def test_str(self):
        str1 = '  x1    x2    x3\n----  ----  ----\n   0     1     2\n   3     4     '\
               '5\n   6     7     8\n   9    10    11'
        str2 = '  x1    x2    5    (1, 2)\n----  ----  ---  --------\n   0     1    2'\
               '         3\n   4     5    6         7\n   8     9   10        11\n  12'\
               '    13   14        15\n  16    17   18        19'

        self.assertEqual(self.phi1.__str__(), str1)
        self.assertEqual(self.phi2.__str__(), str2)

    def test_eq(self):
        self.assertNotEqual(self.phi1, self.phi2)
        self.assertNotEqual(self.phi1, 'abcd')
        self.assertNotEqual(self.phi1, 123)
        self.assertNotEqual(self.phi2, ('x1', 123))

        self.assertEqual(self.phi2, self.phi3)
        self.assertEqual(self.phi2, self.phi4)
        self.assertEqual(self.phi3, self.phi4)
        self.assertEqual(self.phi4, self.phi5)
        self.assertEqual(self.phi2, self.phi5)

    def tearDown(self):
        del self.phi1
        del self.phi2
        del self.phi3
        del self.phi4
        del self.phi5
