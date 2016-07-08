import unittest

import numpy as np
import numpy.testing as np_test
from scipy.stats import multivariate_normal

from pgmpy.factors import JointGaussianDistribution as JGD
from pgmpy.factors import CanonicalFactor


class TestCanonicalFactor(unittest.TestCase):
    def test_class_init(self):
        phi = CanonicalFactor(['x1', ('y', 'z'), 'x3'],
                              np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
                              np.array([[1], [4.7], [-1]]), -2)
        self.assertEqual(phi.variables, ['x1', ('y', 'z'), 'x3'])
        np_test.assert_array_equal(phi.K, np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float))
        np_test.assert_array_equal(phi.h, np.array([[1], [4.7], [-1]], dtype=float))
        self.assertEqual(phi.g, -2)
        self.assertEqual(phi.pdf, None)

        phi = CanonicalFactor(['x1', ('y', 'z'), 'x3'],
                              np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
                              np.array([1, 4.7, -1]), -2)
        self.assertEqual(phi.variables, ['x1', ('y', 'z'), 'x3'])
        np_test.assert_array_equal(phi.K, np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float))
        np_test.assert_array_equal(phi.h, np.array([[1], [4.7], [-1]], dtype=float))
        self.assertEqual(phi.g, -2)
        self.assertEqual(phi.pdf, None)

        phi = CanonicalFactor(['x'], [[1]], [0], 1)
        self.assertEqual(phi.variables, ['x'])
        np_test.assert_array_equal(phi.K, np.array([[1]], dtype=float))
        np_test.assert_array_equal(phi.h, np.array([[0]], dtype=float))
        self.assertEqual(phi.g, 1)

    def test_class_init_valueerror(self):
        self.assertRaises(ValueError, CanonicalFactor, ['x1', 'x2', 'x3'],
                         np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
                         np.array([1, 2]), 7)
        self.assertRaises(ValueError, CanonicalFactor, ['x1', 'x2', 'x3'],
                         np.array([[1.1, -1, 0], [-1, 4], [0, -2, 4]]),
                         np.array([1, 2, 3]), 7)
        self.assertRaises(ValueError, CanonicalFactor, ['x1', 'x2', 'x3'],
                         np.array([[1.1, -1, 0], [0, -2, 4]]),
                         np.array([1, 2, 3]), 7)
        self.assertRaises(ValueError, CanonicalFactor, ['x1', 'x3'],
                         np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
                         np.array([1, 2, 3]), 7)
