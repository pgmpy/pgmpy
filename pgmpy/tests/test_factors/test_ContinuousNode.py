import unittest

import numpy as np
import numpy.testing as np_test
from scipy.stats import norm, t, gamma

from pgmpy.factors import ContinuousNode


class TestContinuousNodeInit(unittest.TestCase):
    def test_class_init1(self):
        custom_pdf = lambda x : 0.5 if x > -1 and x < 1 else 0
        std_normal_pdf = lambda x : np.exp(-x*x/2) / (np.sqrt(2*np.pi))
        exp_pdf = lambda x: 2*np.exp(-2*x) if x>=0 else 0

        custom_node1 = ContinuousNode(custom_pdf)
        custom_node2 = ContinuousNode(custom_pdf, -1, 1)

        normal_node1 = ContinuousNode(std_normal_pdf)
        normal_node2 = ContinuousNode(std_normal_pdf, -5, 5)

        exp_node1 = ContinuousNode(exp_pdf)
        exp_node2 = ContinuousNode(exp_pdf, 0, 10)

        self.assertEqual(custom_node1.pdf, custom_pdf)
        self.assertEqual(custom_node2.pdf, custom_pdf)
        self.assertEqual(normal_node1.pdf, std_normal_pdf)
        self.assertEqual(normal_node1.pdf, std_normal_pdf)
        self.assertEqual(exp_node1.pdf, exp_pdf)
        self.assertEqual(exp_node2.pdf, exp_pdf)

    def test_class_init2(self):
        # A normal random varible with mean=1 and variance=0.5
        norm_rv = norm(1,0.5)
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

