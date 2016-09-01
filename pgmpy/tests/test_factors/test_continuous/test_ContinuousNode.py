from __future__ import division
import unittest

import numpy as np
from scipy.stats import norm, t, gamma

from pgmpy.factors.continuous import ContinuousNode
from pgmpy.factors.continuous import BaseDiscretizer
from pgmpy.factors.continuous import RoundingDiscretizer
from pgmpy.factors.continuous import UnbiasedDiscretizer


class TestContinuousNodeInit(unittest.TestCase):
    def custom_pdf(self, x):
        return 0.5 if (x > -1 and x < 1) else 0

    def std_normal_pdf(self, x):
        return np.exp(-x * x / 2) / (np.sqrt(2 * np.pi))

    def exp_pdf(self, x):
        return 2 * np.exp(-2 * x) if x >= 0 else 0

    def test_class_init1(self):
        custom_node1 = ContinuousNode(self.custom_pdf)
        custom_node2 = ContinuousNode(self.custom_pdf, -1, 1)

        normal_node1 = ContinuousNode(self.std_normal_pdf)
        normal_node2 = ContinuousNode(self.std_normal_pdf, -5, 5)

        exp_node1 = ContinuousNode(self.exp_pdf)
        exp_node2 = ContinuousNode(self.exp_pdf, 0, 10)

        self.assertEqual(custom_node1.pdf, self.custom_pdf)
        self.assertEqual(custom_node2.pdf, self.custom_pdf)
        self.assertEqual(normal_node1.pdf, self.std_normal_pdf)
        self.assertEqual(normal_node2.pdf, self.std_normal_pdf)
        self.assertEqual(exp_node1.pdf, self.exp_pdf)
        self.assertEqual(exp_node2.pdf, self.exp_pdf)

    def test_class_init2(self):
        # A normal random varible with mean=1 and variance=0.5
        norm_rv = norm(1, 0.5)
        # A t random variable with df=7
        t7_rv = t(7)
        # A gamma random variable with a=3
        gamma_rv = gamma(3)

        normal_node = ContinuousNode(norm_rv.pdf)
        t7_node = ContinuousNode(t7_rv.pdf)
        gamma_node = ContinuousNode(gamma_rv.pdf)

        self.assertEqual(normal_node.pdf, norm_rv.pdf)
        self.assertEqual(t7_node.pdf, t7_rv.pdf)
        self.assertEqual(gamma_node.pdf, gamma_rv.pdf)


class TestContinuousNodeDiscretize(unittest.TestCase):
    def setUp(self):
        self.custom_pdf = lambda x: 0.5 if x > -1 and x < 1 else 0
        self.std_normal_pdf = lambda x: np.exp(-x*x/2) / (np.sqrt(2*np.pi))
        self.exp_pdf = lambda x: 2*np.exp(-2*x) if x >= 0 else 0

        self.custom_node1 = ContinuousNode(self.custom_pdf)
        self.custom_node2 = ContinuousNode(self.custom_pdf, -1, 1)

        self.normal_node1 = ContinuousNode(self.std_normal_pdf)
        self.normal_node2 = ContinuousNode(self.std_normal_pdf, -5, 5)

        self.exp_node1 = ContinuousNode(self.exp_pdf)
        self.exp_node2 = ContinuousNode(self.exp_pdf, lb=0)

        self.discretizer1 = UnbiasedDiscretizer(self.custom_node1, -1, 1, 8).get_discrete_values()
        self.discretizer2 = RoundingDiscretizer(self.custom_node1, -1, 1, 8).get_discrete_values()

        self.discretizer3 = UnbiasedDiscretizer(self.normal_node1, -5, 5, 10).get_discrete_values()
        self.discretizer4 = RoundingDiscretizer(self.normal_node1, -5, 5, 10).get_discrete_values()

        self.discretizer5 = UnbiasedDiscretizer(self.exp_node2, 0, 5, 5).get_discrete_values()
        self.discretizer6 = RoundingDiscretizer(self.exp_node2, 0, 5, 5).get_discrete_values()

        self.discretizer7 = UnbiasedDiscretizer(self.custom_node2, -1, 1, 8).get_discrete_values()
        self.discretizer8 = RoundingDiscretizer(self.custom_node2, -1, 1, 8).get_discrete_values()

        self.discretizer9 = UnbiasedDiscretizer(self.normal_node2, -5, 5, 10).get_discrete_values()
        self.discretizer10 = RoundingDiscretizer(self.normal_node2, -5, 5, 10).get_discrete_values()

        self.discretizer11 = UnbiasedDiscretizer(self.exp_node1, 0, 5, 5).get_discrete_values()
        self.discretizer12 = RoundingDiscretizer(self.exp_node1, 0, 5, 5).get_discrete_values()

        class CustomDiscretizer(BaseDiscretizer):
            def get_discrete_values(self):
                step = (self.high - self.low) / self.cardinality
                function = lambda x: self.factor.cdf(x+step) - self.factor.cdf(x)
                discrete_values = [function(i) for i in np.arange(self.low, self.high, step)]
                return discrete_values

        self.CustomDiscretizer = CustomDiscretizer

        self.discretizer13 = CustomDiscretizer(self.custom_node2, -1, 1, 8).get_discrete_values()
        self.discretizer14 = CustomDiscretizer(self.normal_node1, -5, 5, 10).get_discrete_values()
        self.discretizer15 = CustomDiscretizer(self.exp_node2, 0, 5, 5).get_discrete_values()

    def test_discretize_type_error(self):
        self.assertRaises(TypeError, self.custom_node1.discretize, 'Unbiased', -1, 1, 8)
        self.assertRaises(TypeError, self.custom_node1.discretize, -1, 1, 8)
        self.assertRaises(TypeError, self.custom_node1.discretize, -1, 1, 8)
        self.assertRaises(TypeError, self.custom_node1.discretize, self.discretizer1, -1, 1, 8)
        self.assertRaises(TypeError, self.custom_node1.discretize, self.discretizer2, -1, 1, 8)

        self.assertRaises(TypeError, self.normal_node1.discretize, 'Unbiased', -5, 5, 10)
        self.assertRaises(TypeError, self.normal_node1.discretize, -5, 5, 10)
        self.assertRaises(TypeError, self.normal_node1.discretize, 1, -5, 5, 10)
        self.assertRaises(TypeError, self.normal_node1.discretize, self.discretizer3, -5, 5, 10)
        self.assertRaises(TypeError, self.normal_node1.discretize, self.discretizer4, -5, 5, 10)

        self.assertRaises(TypeError, self.exp_node2.discretize, 'Unbiased', -5, 5, 10)
        self.assertRaises(TypeError, self.exp_node2.discretize, -5, 5, 10)
        self.assertRaises(TypeError, self.exp_node2.discretize, 1, -5, 5, 10)
        self.assertRaises(TypeError, self.exp_node2.discretize, self.discretizer5, 0, 5, 5)
        self.assertRaises(TypeError, self.exp_node2.discretize, self.discretizer6, 0, 5, 5)

        self.assertRaises(TypeError, self.custom_node2.discretize, self.discretizer13, -1, 1, 8)
        self.assertRaises(TypeError, self.normal_node1.discretize, self.discretizer14, -5, 5, 10)
        self.assertRaises(TypeError, self.exp_node2.discretize, self.discretizer15, 0, 5, 5)

    def test_discretize(self):
        self.assertEqual(self.custom_node1.discretize(UnbiasedDiscretizer, -1, 1, 8), self.discretizer1)
        self.assertEqual(self.custom_node1.discretize(RoundingDiscretizer, -1, 1, 8), self.discretizer2)
        self.assertEqual(self.custom_node2.discretize(self.CustomDiscretizer, -1, 1, 8), self.discretizer13)
        self.assertEqual(self.custom_node2.discretize(UnbiasedDiscretizer, -1, 1, 8), self.discretizer7)
        self.assertEqual(self.custom_node2.discretize(RoundingDiscretizer, -1, 1, 8), self.discretizer8)

        self.assertEqual(self.normal_node1.discretize(UnbiasedDiscretizer, -5, 5, 10), self.discretizer3)
        self.assertEqual(self.normal_node1.discretize(RoundingDiscretizer, -5, 5, 10), self.discretizer4)
        self.assertEqual(self.normal_node1.discretize(self.CustomDiscretizer, -5, 5, 10), self.discretizer14)
        self.assertEqual(self.normal_node2.discretize(UnbiasedDiscretizer, -5, 5, 10), self.discretizer9)
        self.assertEqual(self.normal_node2.discretize(RoundingDiscretizer, -5, 5, 10), self.discretizer10)

        self.assertEqual(self.exp_node2.discretize(UnbiasedDiscretizer, 0, 5, 5), self.discretizer5)
        self.assertEqual(self.exp_node2.discretize(RoundingDiscretizer, 0, 5, 5), self.discretizer6)
        self.assertEqual(self.exp_node2.discretize(self.CustomDiscretizer, 0, 5, 5), self.discretizer15)
        self.assertEqual(self.exp_node1.discretize(UnbiasedDiscretizer, 0, 5, 5), self.discretizer11)
        self.assertEqual(self.exp_node1.discretize(RoundingDiscretizer, 0, 5, 5), self.discretizer12)

    def tearDown(self):
        del self.custom_pdf
        del self.std_normal_pdf
        del self.exp_pdf
        del self.custom_node1
        del self.custom_node2
        del self.exp_node1
        del self.exp_node2
        del self.normal_node1
        del self.normal_node2
        del self.discretizer1
        del self.discretizer2
        del self.discretizer3
        del self.discretizer4
        del self.discretizer5
        del self.discretizer6
        del self.discretizer7
        del self.discretizer8
        del self.discretizer9
        del self.discretizer10
        del self.discretizer11
        del self.discretizer12
        del self.discretizer13
        del self.discretizer14
        del self.discretizer15
