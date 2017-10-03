import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.distributions import GaussianDistribution as GD


class TestGDInit(unittest.TestCase):
    def test_class_init(self):
        phi1 = GD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
                  np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi1.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi1.mean, np.asarray([[1], [-3], [4]], dtype=float))
        np_test.assert_array_equal(phi1.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi1._precision_matrix, None)

        phi2 = GD(['x1', 'x2', 'x3'], [1, 2, 5],
                  np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi2.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(phi2.mean, np.asarray([[1], [2], [5]], dtype=float))
        np_test.assert_array_equal(phi2.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi2._precision_matrix, None)

        phi3 = GD(['x'], [0], [[1]])
        self.assertEqual(phi3.variables, ['x'])
        np_test.assert_array_equal(phi3.mean, np.asarray([[0]], dtype=float))
        np_test.assert_array_equal(phi3.covariance, np.asarray([[1]], dtype=float))
        self.assertEqual(phi3._precision_matrix, None)

        phi1 = GD(['1', 2, (1, 2, 'x')],
                  np.array([[1], [-3], [4]]),
                  np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi1.variables, ['1', 2, (1, 2, 'x')])
        np_test.assert_array_equal(phi1.mean, np.asarray([[1], [-3], [4]], dtype=float))
        np_test.assert_array_equal(phi1.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi1._precision_matrix, None)

        phi2 = GD(['1', 7, (1, 2, 'x')], [1, 2, 5],
                  np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertEqual(phi2.variables, ['1', 7, (1, 2, 'x')])
        np_test.assert_array_equal(phi2.mean, np.asarray([[1], [2], [5]], dtype=float))
        np_test.assert_array_equal(phi2.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(phi2._precision_matrix, None)

        phi3 = GD([23], [0], [[1]])
        self.assertEqual(phi3.variables, [23])
        np_test.assert_array_equal(phi3.mean, np.asarray([[0]], dtype=float))
        np_test.assert_array_equal(phi3.covariance, np.asarray([[1]], dtype=float))
        self.assertEqual(phi3._precision_matrix, None)

    def test_class_init_valueerror(self):
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2'], [1, -3, 4],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [[1, -3, 4]],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [[1], [-3]],
                          np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))

        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1, -3, 4],
                          np.array([[4, 2, -2], [2, 5, -5]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1, -3, 4],
                          np.array([[4, 2, -2]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[4, 2], [2, 5], [-2, -5]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[4], [2], [-2]]))
        self.assertRaises(ValueError, GD, ['x1', 'x2', 'x3'], [1, -3],
                          np.array([[-2]]))


class TestJGDMethods(unittest.TestCase):
    def setUp(self):
        self.phi1 = GD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
                       np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        self.phi2 = GD(['x'], [0], [[1]])
        self.phi3 = self.phi1.copy()

    def test_precision_matrix(self):
        self.assertEqual(self.phi1._precision_matrix, None)
        np_test.assert_almost_equal(self.phi1.precision_matrix,
                                    np.array([[0.3125, -0.125, 0],
                                             [-0.125, 0.5833333, 0.3333333],
                                             [0, 0.3333333, 0.3333333]]))
        np_test.assert_almost_equal(self.phi1._precision_matrix,
                                    np.array([[0.3125, -0.125, 0],
                                             [-0.125, 0.5833333, 0.3333333],
                                             [0, 0.3333333, 0.3333333]]))

        self.assertEqual(self.phi2._precision_matrix, None)
        np_test.assert_almost_equal(self.phi2.precision_matrix, np.array([[1]]))
        np_test.assert_almost_equal(self.phi2._precision_matrix, np.array([[1]]))

    def test_marginalize(self):
        phi = self.phi1.marginalize(['x3'], inplace=False)
        self.assertEqual(phi.variables, ['x1', 'x2'])
        np_test.assert_array_equal(phi.mean, np.asarray([[1], [-3]], dtype=float))
        np_test.assert_array_equal(phi.covariance,
                                   np.asarray([[4, 2], [2, 5]], dtype=float))
        self.assertEqual(phi._precision_matrix, None)

        phi = self.phi1.marginalize(['x3', 'x2'], inplace=False)
        self.assertEqual(phi.variables, ['x1'])
        np_test.assert_array_equal(phi.mean, np.asarray([[1]], dtype=float))
        np_test.assert_array_equal(phi.covariance,
                                   np.asarray([[4]], dtype=float))
        self.assertEqual(phi._precision_matrix, None)

        self.phi1.marginalize(['x3'])
        self.assertEqual(self.phi1.variables, ['x1', 'x2'])
        np_test.assert_array_equal(self.phi1.mean, np.asarray([[1], [-3]], dtype=float))
        np_test.assert_array_equal(self.phi1.covariance,
                                   np.asarray([[4, 2], [2, 5]], dtype=float))
        self.assertEqual(self.phi1._precision_matrix, None)

        self.phi1 = self.phi3
        self.phi1.marginalize(['x3', 'x2'])
        self.assertEqual(self.phi1.variables, ['x1'])
        np_test.assert_array_equal(self.phi1.mean, np.asarray([[1]], dtype=float))
        np_test.assert_array_equal(self.phi1.covariance,
                                   np.asarray([[4]], dtype=float))
        self.assertEqual(self.phi1._precision_matrix, None)

        self.phi1 = self.phi3

    def test_copy(self):
        copy_phi1 = self.phi1.copy()
        self.assertEqual(copy_phi1.variables, self.phi1.variables)
        np_test.assert_array_equal(copy_phi1.mean, self.phi1.mean)
        np_test.assert_array_equal(copy_phi1.covariance, self.phi1.covariance)
        np_test.assert_array_equal(copy_phi1._precision_matrix, self.phi1._precision_matrix)

        copy_phi1.marginalize(['x3'])
        self.assertEqual(self.phi1.variables, ['x1', 'x2', 'x3'])
        np_test.assert_array_equal(self.phi1.mean, np.asarray([[1], [-3], [4]], dtype=float))
        np_test.assert_array_equal(self.phi1.covariance,
                                   np.asarray([[4, 2, -2], [2, 5, -5], [-2, -5, 8]], dtype=float))
        self.assertEqual(self.phi1._precision_matrix, None)

        self.phi1.marginalize(['x2'])
        self.assertEqual(copy_phi1.variables, ['x1', 'x2'])
        np_test.assert_array_equal(copy_phi1.mean, np.asarray([[1], [-3]], dtype=float))
        np_test.assert_array_equal(copy_phi1.covariance,
                                   np.asarray([[4, 2], [2, 5]], dtype=float))
        self.assertEqual(copy_phi1._precision_matrix, None)

        self.phi1 = self.phi3

    def test_assignment(self):
        np_test.assert_almost_equal(self.phi1.assignment(*[1, 2, 3]), 2.797826e-05)
        np_test.assert_almost_equal(self.phi1.assignment(*[[1, 2, 3], [0, 0, 0]]),
                                    np.array([2.79782602e-05, 1.48056313e-03]))
        np_test.assert_almost_equal(self.phi2.assignment(0), 0.3989422804)
        np_test.assert_almost_equal(self.phi2.assignment(*[0, 1, -1]),
                                    np.array([0.39894228, 0.24197072, 0.24197072]))

    def test_reduce(self):
        phi = self.phi1.reduce([('x1', 7)], inplace=False)
        self.assertEqual(phi.variables, ['x2', 'x3'])
        np_test.assert_array_equal(phi.mean, np.asarray([[0], [1]], dtype=float))
        np_test.assert_array_equal(phi.covariance,
                                   np.asarray([[4, -4], [-4, 7]], dtype=float))
        self.assertEqual(phi._precision_matrix, None)

        phi = self.phi1.reduce([('x1', 3), ('x2', 1)], inplace=False)
        self.assertEqual(phi.variables, ['x3'])
        np_test.assert_array_equal(phi.mean, np.array([[0]], dtype=float))
        np_test.assert_array_equal(phi.covariance,
                                   np.asarray([[3]], dtype=float))
        self.assertEqual(phi._precision_matrix, None)

        self.phi1.reduce([('x1', 7)])
        self.assertEqual(self.phi1.variables, ['x2', 'x3'])
        np_test.assert_array_equal(self.phi1.mean, np.asarray([[0], [1]], dtype=float))
        np_test.assert_array_equal(self.phi1.covariance,
                                   np.asarray([[4, -4], [-4, 7]], dtype=float))
        self.assertEqual(self.phi1._precision_matrix, None)

        self.phi1 = self.phi3.copy()
        self.phi1.reduce([('x1', 3), ('x2', 1)])
        self.assertEqual(self.phi1.variables, ['x3'])
        np_test.assert_array_equal(self.phi1.mean, np.asarray([[0]], dtype=float))
        np_test.assert_array_equal(self.phi1.covariance,
                                   np.asarray([[3]], dtype=float))
        self.assertEqual(self.phi1._precision_matrix, None)

        self.phi1 = self.phi3.copy()
        self.phi1.reduce([('x2', 1), ('x1', 3)])
        self.assertEqual(self.phi1.variables, ['x3'])
        np_test.assert_array_equal(self.phi1.mean, np.asarray([[0]], dtype=float))
        np_test.assert_array_equal(self.phi1.covariance,
                                   np.asarray([[3]], dtype=float))
        self.assertEqual(self.phi1._precision_matrix, None)

    def test_normalize(self):
        phi = self.phi1.copy()
        phi.normalize()
        self.assertEqual(self.phi1.variables, phi.variables)
        np_test.assert_array_equal(self.phi1.mean, phi.mean)
        np_test.assert_array_equal(self.phi1.covariance, phi.covariance)
        self.assertEqual(self.phi1._precision_matrix, phi._precision_matrix)

        phi = self.phi1.normalize(inplace=False)
        self.assertEqual(self.phi1.variables, phi.variables)
        np_test.assert_array_equal(self.phi1.mean, phi.mean)
        np_test.assert_array_equal(self.phi1.covariance, phi.covariance)
        self.assertEqual(self.phi1._precision_matrix, phi._precision_matrix)

    def test_product(self):
        pass

    def test_divide(self):
        pass

    def test_eq(self):
        pass

    def test_repr(self):
        pass

    def tearDown(self):
        del self.phi1
        del self.phi2
        del self.phi3
