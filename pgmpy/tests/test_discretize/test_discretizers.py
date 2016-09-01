import unittest

import numpy as np
import numpy.testing as np_test
from scipy.stats import gamma, expon

from pgmpy.factors.continuous import ContinuousNode
from pgmpy.discretize import BaseDiscretizer
from pgmpy.discretize import RoundingDiscretizer
from pgmpy.discretize import UnbiasedDiscretizer


class TestBaseDiscretizer(unittest.TestCase):
    def setUp(self):
        self.normal_pdf = lambda x: np.exp(-x*x/2) / (np.sqrt(2*np.pi))
        gamma_rv = gamma(3)
        self.gamma_pdf = gamma_rv.pdf
        exp_rv = expon(7)
        self.exp_pdf = exp_rv.pdf

        class ChildDiscretizer(BaseDiscretizer):
            def get_discrete_values(self):
                pass

        self.normal_factor = ContinuousNode(self.normal_pdf)
        self.gamma_factor = ContinuousNode(self.gamma_pdf, lb=0)
        self.exp_factor = ContinuousNode(self.exp_pdf, lb=0)

        self.normal_discretizer = ChildDiscretizer(self.normal_factor, -10, 10, 20)
        self.gamma_discretizer = ChildDiscretizer(self.gamma_factor, 0, 10, 10)
        self.exp_discretizer = ChildDiscretizer(self.exp_factor, 0, 5, 10)

    def test_base_init(self):
        self.assertEqual(self.normal_discretizer.factor, self.normal_factor)
        self.assertEqual(self.normal_discretizer.low, -10)
        self.assertEqual(self.normal_discretizer.high, 10)
        self.assertEqual(self.normal_discretizer.cardinality, 20)

        self.assertEqual(self.gamma_discretizer.factor, self.gamma_factor)
        self.assertEqual(self.gamma_discretizer.low, 0)
        self.assertEqual(self.gamma_discretizer.high, 10)
        self.assertEqual(self.gamma_discretizer.cardinality, 10)

        self.assertEqual(self.exp_discretizer.factor, self.exp_factor)
        self.assertEqual(self.exp_discretizer.low, 0)
        self.assertEqual(self.exp_discretizer.high, 5)
        self.assertEqual(self.exp_discretizer.cardinality, 10)

    def test_get_labels(self):
        o1 = ['x=-10.0', 'x=-9.0', 'x=-8.0', 'x=-7.0', 'x=-6.0', 'x=-5.0', 'x=-4.0', 'x=-3.0',
              'x=-2.0', 'x=-1.0', 'x=0.0', 'x=1.0', 'x=2.0', 'x=3.0', 'x=4.0', 'x=5.0', 'x=6.0',
              'x=7.0', 'x=8.0', 'x=9.0']
        o2 = ['x=0.0', 'x=1.0', 'x=2.0', 'x=3.0', 'x=4.0', 'x=5.0', 'x=6.0', 'x=7.0', 'x=8.0', 'x=9.0']
        o3 = ['x=0.0', 'x=0.5', 'x=1.0', 'x=1.5', 'x=2.0', 'x=2.5', 'x=3.0', 'x=3.5', 'x=4.0', 'x=4.5']

        self.assertListEqual(self.normal_discretizer.get_labels(), o1)
        self.assertListEqual(self.gamma_discretizer.get_labels(), o2)
        self.assertListEqual(self.exp_discretizer.get_labels(), o3)

    def tearDown(self):
        del self.normal_pdf
        del self.gamma_pdf
        del self.exp_pdf
        del self.normal_factor
        del self.gamma_factor
        del self.exp_factor
        del self.normal_discretizer
        del self.gamma_discretizer
        del self.exp_discretizer


class TestRoundingDiscretizer(unittest.TestCase):
    def setUp(self):
        self.normal_pdf = lambda x: np.exp(-x*x/2) / (np.sqrt(2*np.pi))
        gamma_rv = gamma(3)
        self.gamma_pdf = gamma_rv.pdf
        exp_rv = expon()
        self.exp_pdf = exp_rv.pdf

        self.normal_factor = ContinuousNode(self.normal_pdf)
        self.gamma_factor = ContinuousNode(self.gamma_pdf, lb=0)
        self.exp_factor = ContinuousNode(self.exp_pdf, lb=0)

        self.normal_discretizer = RoundingDiscretizer(self.normal_factor, -5, 5, 10)
        self.normal_discretizer2 = RoundingDiscretizer(self.normal_factor, -3, 3, 10)
        self.gamma_discretizer = RoundingDiscretizer(self.gamma_factor, 0, 5, 5)
        self.exp_discretizer = RoundingDiscretizer(self.exp_factor, 0, 5, 10)

    def test_get_discrete_values(self):
        # The output for the get_discrete_values method has been cross checked
        # using discretize {actuar} package in R, assuming that it gives correct results.
        # The required R commands to reproduce the results have also been added.

        # library(actuar);discretize(pnorm(x), method = "rounding", from = -5, to = 5, step = 1)
        normal_desired_op = np.array([3.111022e-06, 2.292314e-04, 5.977036e-03, 6.059754e-02, 2.417303e-01,
                                      3.829249e-01, 2.417303e-01, 6.059754e-02, 5.977036e-03, 2.292314e-04])
        normal_obtained_op = np.array(self.normal_discretizer.get_discrete_values())
        np_test.assert_almost_equal(normal_desired_op, normal_obtained_op)

        # library(actuar);discretize(pnorm(x), method = "rounding", from = -3, to = 3, step = 6/10)
        normal_desired_op2 = np.array([0.002117076, 0.014397447, 0.048942781, 0.117252924, 0.198028452,
                                       0.235822844, 0.198028452, 0.117252924, 0.048942781, 0.014397447])
        normal_obtained_op2 = np.array(self.normal_discretizer2.get_discrete_values())
        np_test.assert_almost_equal(normal_desired_op2, normal_obtained_op2)

        # library(actuar);discretize(pgamma(x, 3), method = "rounding", from = 0, to = 5, step = 1)
        gamma_desired_op = np.array([0.01438768, 0.17676549, 0.26503371, 0.22296592, 0.14726913])
        gamma_obtained_op = np.array(self.gamma_discretizer.get_discrete_values())
        np_test.assert_almost_equal(gamma_desired_op, gamma_obtained_op)

        # library(actuar);discretize(pexp(x), method = "rounding", from = 0, to = 5, step = 0.5)
        exp_desired_op = np.array([0.221199217, 0.306434230, 0.185861756, 0.112730853, 0.068374719,
                                   0.041471363, 0.025153653, 0.015256462, 0.009253512, 0.005612539])
        exp_obtained_op = np.array(self.exp_discretizer.get_discrete_values())
        np_test.assert_almost_equal(exp_desired_op, exp_obtained_op)

    def tearDown(self):
        del self.normal_pdf
        del self.gamma_pdf
        del self.exp_pdf
        del self.normal_factor
        del self.gamma_factor
        del self.exp_factor
        del self.normal_discretizer
        del self.gamma_discretizer
        del self.exp_discretizer


class TestUnbiasedDiscretizer(unittest.TestCase):
    def setUp(self):
        gamma_rv = gamma(3)
        self.gamma_pdf = gamma_rv.pdf
        exp_rv = expon()
        self.exp_pdf = exp_rv.pdf

        self.gamma_factor = ContinuousNode(self.gamma_pdf, lb=0)
        self.exp_factor = ContinuousNode(self.exp_pdf, lb=0)

        self.gamma_discretizer = UnbiasedDiscretizer(self.gamma_factor, 0, 5, 5)
        self.exp_discretizer = UnbiasedDiscretizer(self.exp_factor, 0, 5, 10)

    def test_get_discrete_values(self):
        # The output for the get_discrete_values method has been cross checked
        # using discretize {actuar} package in R, assuming that it gives correct results.
        # The required R commands to reproduce the results have also been added.

        # library(actuar);discretize(pgamma(x, 3), method = "unbiased", lev = levgamma(x, 3), from = 0, to = 5, step = 5/4)
        gamma_desired_op = np.array([0.03968660, 0.25118328, 0.30841001, 0.20833784, 0.06773025])
        gamma_obtained_op = np.array(self.gamma_discretizer.get_discrete_values())
        np_test.assert_almost_equal(gamma_desired_op, gamma_obtained_op)

        # library(actuar);discretize(pexp(x), method = "unbiased", lev = levexp(x), from = 0, to = 5, step = 5/9)
        exp_desired_op = np.array([0.232756157, 0.327035063, 0.187637486, 0.107657650, 0.061768945, 0.035440143,
                                   0.020333903, 0.011666647, 0.006693778, 0.002272280])
        exp_obtained_op = np.array(self.exp_discretizer.get_discrete_values())
        np_test.assert_almost_equal(exp_desired_op, exp_obtained_op)

    def tearDown(self):
        del self.gamma_pdf
        del self.exp_pdf
        del self.gamma_factor
        del self.exp_factor
        del self.gamma_discretizer
        del self.exp_discretizer
