import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.factors.continuous import LinearGaussianCPD


class TestLGCPD(unittest.TestCase):
    def test_class_init(self):
        beta = np.array([7, 13])
        std = np.array([[4, 3], [3, 6]])

        cpd1 = LinearGaussianCPD("Y", beta=beta, std=std, evidence=["X1", "X2"])
        self.assertEqual(cpd1.variable, "Y")
        self.assertEqual(cpd1.evidence, ["X1", "X2"])

    def test_str(self):
        cpd1 = LinearGaussianCPD("x", [0.23], 0.56)
        cpd2 = LinearGaussianCPD("y", [0.67, 1, 4.56, 8], 2, ["x1", "x2", "x3"])
        self.assertEqual(cpd1.__str__(), "P(x) = N(0.23; 0.56)")
        self.assertEqual(
            cpd2.__str__(),
            "P(y | x1, x2, x3) = N(1.0*x1 + 4.56*x2 + 8.0*x3 + 0.67; 2)",
        )

    def test_get_random(self):
        cpd_random = LinearGaussianCPD.get_random("x", ["x1", "x2", "x3"], 0.23, 0.56)
        self.assertIn("P(x | x1, x2, x3) = N(", cpd_random.__str__())
