import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors import JointGaussianDistribution as JGD


class TestJGDInit(unittest.TestCase):
    def test_class_init(self):
        phi1 = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
                    np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi1.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi1.mean, np.asarray([[1], [-3], [4]], dtype=float))
        np_test.assert_array_equal(phi1.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi1._precision_matrix, None)

        phi2 = JGD(['x1', 'x2', 'x3'], [1, 2, 5],
                    np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi2.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi2.mean, np.asarray([[1], [2], [5]], dtype=float))
        np_test.assert_array_equal(phi2.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi2._precision_matrix, None)

        phi3 = JGD(['x'], [0], [[1]])
        self.assertEqual(phi3.variables, ['x'])
        np_test.assert_array_equal(phi3.mean, np.asarray([[0],], dtype=float))
        np_test.assert_array_equal(phi3.covariance, np.asarray([[1]], dtype=float))
        self.assertEqual(phi3._precision_matrix, None)

        phi1 = JGD(['1', 2, (1, 2, 'x')], np.array([[1], [-3], [4]]),
                    np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi1.variables, ['1', 2, (1, 2, 'x')])
        np_test.assert_array_equal(phi1.mean, np.asarray([[1], [-3], [4]], dtype=float))
        np_test.assert_array_equal(phi1.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi1._precision_matrix, None)

        phi2 = JGD(['1', 7, (1, 2, 'x')], [1, 2, 5],
                    np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi2.variables, ['1', 7, (1, 2, 'x')])
        np_test.assert_array_equal(phi2.mean, np.asarray([[1], [2], [5]], dtype=float))
        np_test.assert_array_equal(phi2.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi2._precision_matrix, None)

        phi3 = JGD([23], [0], [[1]])
        self.assertEqual(phi3.variables, [23])
        np_test.assert_array_equal(phi3.mean, np.asarray([[0],], dtype=float))
        np_test.assert_array_equal(phi3.covariance, np.asarray([[1]], dtype=float))
        self.assertEqual(phi3._precision_matrix, None)

    def test_class_init_valueerror(self):
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2'], [1, -3, 4],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [[1, -3, 4]],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [[1], [-3]],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))

        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1, -3, 4],
                          np.array([[4, 2, -2], [2, 5, -5]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1, -3, 4],
                          np.array([[4, 2, -2]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[4, 2], [2, 5], [-2, -5]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[4], [2], [-2]]))
        self.assertRaises(ValueError, JGD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[-2]]))

