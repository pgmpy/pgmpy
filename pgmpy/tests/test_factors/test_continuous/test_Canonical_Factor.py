import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.distributions import GaussianDistribution as JGD
from pgmpy.factors.continuous import CanonicalDistribution


class TestCanonicalFactor(unittest.TestCase):
    def test_class_init(self):
        phi = CanonicalDistribution(
            ["x1", ("y", "z"), "x3"],
            np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
            np.array([[1], [4.7], [-1]]),
            -2,
        )
        self.assertEqual(phi.variables, ["x1", ("y", "z"), "x3"])
        np_test.assert_array_equal(
            phi.K, np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float)
        )
        np_test.assert_array_equal(phi.h, np.array([[1], [4.7], [-1]], dtype=float))
        self.assertEqual(phi.g, -2)

        phi = CanonicalDistribution(
            ["x1", ("y", "z"), "x3"],
            np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
            np.array([1, 4.7, -1]),
            -2,
        )
        self.assertEqual(phi.variables, ["x1", ("y", "z"), "x3"])
        np_test.assert_array_equal(
            phi.K, np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float)
        )
        np_test.assert_array_equal(phi.h, np.array([[1], [4.7], [-1]], dtype=float))
        self.assertEqual(phi.g, -2)

        phi = CanonicalDistribution(["x"], [[1]], [0], 1)
        self.assertEqual(phi.variables, ["x"])
        np_test.assert_array_equal(phi.K, np.array([[1]], dtype=float))
        np_test.assert_array_equal(phi.h, np.array([[0]], dtype=float))
        self.assertEqual(phi.g, 1)

    def test_class_init_valueerror(self):
        self.assertRaises(
            ValueError,
            CanonicalDistribution,
            ["x1", "x2", "x3"],
            np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float),
            np.array([1, 2], dtype=float),
            7,
        )
        self.assertRaises(
            ValueError,
            CanonicalDistribution,
            ["x1", "x2", "x3"],
            np.array([[1.1, -1, 0], [-1, 4], [0, -2, 4]], dtype=object),
            np.array([1, 2, 3], dtype=float),
            7,
        )
        self.assertRaises(
            ValueError,
            CanonicalDistribution,
            ["x1", "x2", "x3"],
            np.array([[1.1, -1, 0], [0, -2, 4]], dtype=float),
            np.array([1, 2, 3], dtype=float),
            7,
        )
        self.assertRaises(
            ValueError,
            CanonicalDistribution,
            ["x1", "x3"],
            np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float),
            np.array([1, 2, 3], dtype=float),
            7,
        )


class TestJGDMethods(unittest.TestCase):
    def setUp(self):
        self.phi1 = CanonicalDistribution(
            ["x1", "x2", "x3"],
            np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
            np.array([[1], [4.7], [-1]]),
            -2,
        )
        self.phi2 = CanonicalDistribution(["x"], [[1]], [0], 1)
        self.phi3 = self.phi1.copy()

        self.gauss_phi1 = JGD(
            ["x1", "x2", "x3"],
            np.array([[3.13043478], [2.44347826], [0.97173913]]),
            np.array(
                [
                    [1.30434783, 0.43478261, 0.2173913],
                    [0.43478261, 0.47826087, 0.23913043],
                    [0.2173913, 0.23913043, 0.36956522],
                ],
                dtype=float,
            ),
        )
        self.gauss_phi2 = JGD(["x"], np.array([0]), np.array([[1]]))

    def test_assignment(self):
        np_test.assert_almost_equal(self.phi1.assignment(1, 2, 3), 0.0007848640)
        np_test.assert_almost_equal(self.phi2.assignment(1.2), 1.323129812337)

    def test_to_joint_gaussian(self):
        jgd1 = self.phi1.to_joint_gaussian()
        jgd2 = self.phi2.to_joint_gaussian()

        self.assertEqual(jgd1.variables, self.gauss_phi1.variables)
        np_test.assert_almost_equal(jgd1.covariance, self.gauss_phi1.covariance)
        np_test.assert_almost_equal(jgd1.mean, self.gauss_phi1.mean)

        self.assertEqual(jgd2.variables, self.gauss_phi2.variables)
        np_test.assert_almost_equal(jgd2.covariance, self.gauss_phi2.covariance)
        np_test.assert_almost_equal(jgd2.mean, self.gauss_phi2.mean)

    def test_reduce(self):
        phi = self.phi1.reduce([("x1", 7)], inplace=False)
        self.assertEqual(phi.variables, ["x2", "x3"])
        np_test.assert_almost_equal(phi.K, np.array([[4.0, -2.0], [-2.0, 4.0]]))
        np_test.assert_almost_equal(phi.h, np.array([[11.7], [-1.0]]))
        np_test.assert_almost_equal(phi.g, -21.95)

        phi = self.phi1.reduce([("x1", 4), ("x2", 1.23)], inplace=False)
        self.assertEqual(phi.variables, ["x3"])
        np_test.assert_almost_equal(phi.K, np.array([[4.0]]))
        np_test.assert_almost_equal(phi.h, np.array([[1.46]]))
        np_test.assert_almost_equal(phi.g, 0.8752)

        self.phi1.reduce([("x1", 7)])
        self.assertEqual(self.phi1.variables, ["x2", "x3"])
        np_test.assert_almost_equal(self.phi1.K, np.array([[4.0, -2.0], [-2.0, 4.0]]))
        np_test.assert_almost_equal(self.phi1.h, np.array([[11.7], [-1.0]]))
        np_test.assert_almost_equal(self.phi1.g, -21.95)

        self.phi1 = self.phi3.copy()
        self.phi1.reduce([("x1", 4), ("x2", 1.23)])
        self.assertEqual(self.phi1.variables, ["x3"])
        np_test.assert_almost_equal(self.phi1.K, np.array([[4.0]]))
        np_test.assert_almost_equal(self.phi1.h, np.array([[1.46]]))
        np_test.assert_almost_equal(self.phi1.g, 0.8752)

        self.phi1 = self.phi3.copy()
        self.phi1.reduce([("x2", 1.23), ("x1", 4)])
        self.assertEqual(self.phi1.variables, ["x3"])
        np_test.assert_almost_equal(self.phi1.K, np.array([[4.0]]))
        np_test.assert_almost_equal(self.phi1.h, np.array([[1.46]]))
        np_test.assert_almost_equal(self.phi1.g, 0.8752)

    def test_marginalize(self):
        phi = self.phi1.marginalize(["x1"], inplace=False)
        self.assertEqual(phi.variables, ["x2", "x3"])
        np_test.assert_almost_equal(phi.K, np.array([[3.090909, -2.0], [-2.0, 4.0]]))
        np_test.assert_almost_equal(phi.h, np.array([[5.6090909], [-1.0]]))
        np_test.assert_almost_equal(phi.g, -0.5787165566)

        phi = self.phi1.marginalize(["x1", "x2"], inplace=False)
        self.assertEqual(phi.variables, ["x3"])
        np_test.assert_almost_equal(phi.K, np.array([[2.70588235]]))
        np_test.assert_almost_equal(phi.h, np.array([[2.62941176]]))
        np_test.assert_almost_equal(phi.g, 39.25598935059)

        self.phi1.marginalize(["x1"])
        self.assertEqual(self.phi1.variables, ["x2", "x3"])
        np_test.assert_almost_equal(
            self.phi1.K, np.array([[3.090909, -2.0], [-2.0, 4.0]])
        )
        np_test.assert_almost_equal(self.phi1.h, np.array([[5.6090909], [-1.0]]))
        np_test.assert_almost_equal(self.phi1.g, -0.5787165566)

        self.phi1 = self.phi3
        self.phi1.marginalize(["x1", "x2"])
        self.assertEqual(self.phi1.variables, ["x3"])
        np_test.assert_almost_equal(self.phi1.K, np.array([[2.70588235]]))
        np_test.assert_almost_equal(self.phi1.h, np.array([[2.62941176]]))
        np_test.assert_almost_equal(self.phi1.g, 39.25598935059)

        self.phi1 = self.phi3

    def test_operate(self):
        phi1 = self.phi1 * CanonicalDistribution(
            ["x2", "x4"], [[1, 2], [3, 4]], [0, 4.56], -6.78
        )
        phi2 = self.phi1 / CanonicalDistribution(
            ["x2", "x3"], [[1, 2], [3, 4]], [0, 4.56], -6.78
        )

        self.assertEqual(phi1.variables, ["x1", "x2", "x3", "x4"])
        np_test.assert_almost_equal(
            phi1.K,
            np.array(
                [
                    [1.1, -1.0, 0.0, 0.0],
                    [-1.0, 5.0, -2.0, 2.0],
                    [0.0, -2.0, 4.0, 0.0],
                    [0.0, 3.0, 0.0, 4.0],
                ]
            ),
        )
        np_test.assert_almost_equal(phi1.h, np.array([[1.0], [4.7], [-1.0], [4.56]]))
        np_test.assert_almost_equal(phi1.g, -8.78)

        self.assertEqual(phi2.variables, ["x1", "x2", "x3"])
        np_test.assert_almost_equal(
            phi2.K, np.array([[1.1, -1.0, 0.0], [-1.0, 3.0, -4.0], [0.0, -5.0, 0.0]])
        )
        np_test.assert_almost_equal(phi2.h, np.array([[1.0], [4.7], [-5.56]]))
        np_test.assert_almost_equal(phi2.g, 4.78)

    def test_copy(self):
        copy_phi1 = self.phi1.copy()
        self.assertEqual(copy_phi1.variables, self.phi1.variables)
        np_test.assert_array_equal(copy_phi1.K, self.phi1.K)
        np_test.assert_array_equal(copy_phi1.h, self.phi1.h)
        np_test.assert_array_equal(copy_phi1.g, self.phi1.g)

        copy_phi1.marginalize(["x1"])
        self.assertEqual(self.phi1.variables, ["x1", "x2", "x3"])
        np_test.assert_array_equal(
            self.phi1.K, np.array([[1.1, -1, 0], [-1, 4, -2], [0, -2, 4]], dtype=float)
        )
        np_test.assert_array_equal(
            self.phi1.h, np.array([[1], [4.7], [-1]], dtype=float)
        )
        self.assertEqual(self.phi1.g, -2)

        self.phi1.marginalize(["x2"])
        self.assertEqual(copy_phi1.variables, ["x2", "x3"])
        np_test.assert_almost_equal(
            copy_phi1.K, np.array([[3.090909, -2.0], [-2.0, 4.0]])
        )
        np_test.assert_almost_equal(copy_phi1.h, np.array([[5.6090909], [-1.0]]))
        np_test.assert_almost_equal(copy_phi1.g, -0.5787165566)

        self.phi1 = self.phi3

    def tearDown(self):
        del self.phi1
        del self.phi2
        del self.phi3
        del self.gauss_phi1
        del self.gauss_phi2
