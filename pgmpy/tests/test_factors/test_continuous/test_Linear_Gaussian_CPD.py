import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.factors.continuous import LinearGaussianCPD


class TestLGCPD(unittest.TestCase):
    def test_class_init(self):
        mu = np.array([7, 13])
        sigma = np.array([[4, 3], [3, 6]])

        cpd1 = LinearGaussianCPD(
            "Y", evidence_mean=mu, evidence_variance=sigma, evidence=["X1", "X2"]
        )
        self.assertEqual(cpd1.variable, "Y")
        self.assertEqual(cpd1.evidence, ["X1", "X2"])

    def test_maximum_likelihood_estimator(self):
        # Obtain the X and Y which are jointly gaussian from the distribution
        # beta = [2, 0.7, 0.3]
        sigma_c = 4

        x_df = pd.read_csv("pgmpy/tests/test_factors/test_continuous/gbn_values_1.csv")

        mu = np.array([7, 13])
        sigma = np.array([[4, 3], [3, 6]])

        cpd1 = LinearGaussianCPD(
            "Y", evidence_mean=mu, evidence_variance=sigma, evidence=["X1", "X2"]
        )
        mean, variance = cpd1.fit(x_df, states=["(Y|X)", "X1", "X2"], estimator="MLE")
        np_test.assert_allclose(mean, [2.361152, 0.693147, 0.276383], rtol=1e-03)
        np_test.assert_allclose(variance, sigma_c, rtol=1e-1)

    @unittest.skip("TODO")
    def test_pdf(self):
        cpd1 = LinearGaussianCPD("x", [0.23], 0.56)
        cpd2 = LinearGaussianCPD("y", [0.67, 1, 4.56, 8], 2, ["x1", "x2", "x3"])
        np_test.assert_almost_equal(cpd1.assignment(1), 0.3139868)
        np_test.assert_almost_equal(cpd2.assignment(1, 1.2, 2.3, 3.4), 1.076e-162)

    @unittest.skip("TODO")
    def test_copy(self):
        cpd = LinearGaussianCPD("y", [0.67, 1, 4.56, 8], 2, ["x1", "x2", "x3"])
        copy = cpd.copy()

        self.assertEqual(cpd.variable, copy.variable)
        self.assertEqual(cpd.beta_0, copy.beta_0)
        self.assertEqual(cpd.variance, copy.variance)
        np_test.assert_array_equal(cpd.beta_vector, copy.beta_vector)
        self.assertEqual(cpd.evidence, copy.evidence)

        cpd.variable = "z"
        self.assertEqual(copy.variable, "y")
        cpd.variance = 0
        self.assertEqual(copy.variance, 2)
        cpd.beta_0 = 1
        self.assertEqual(copy.beta_0, 0.67)
        cpd.evidence = ["p", "q", "r"]
        self.assertEqual(copy.evidence, ["x1", "x2", "x3"])
        cpd.beta_vector = [2, 2, 2]
        np_test.assert_array_equal(copy.beta_vector, [1, 4.56, 8])

        copy = cpd.copy()

        copy.variable = "k"
        self.assertEqual(cpd.variable, "z")
        copy.variance = 0.3
        self.assertEqual(cpd.variance, 0)
        copy.beta_0 = 1.5
        self.assertEqual(cpd.beta_0, 1)
        copy.evidence = ["a", "b", "c"]
        self.assertEqual(cpd.evidence, ["p", "q", "r"])
        copy.beta_vector = [2.2, 2.2, 2.2]
        np_test.assert_array_equal(cpd.beta_vector, [2, 2, 2])

    def test_str(self):
        cpd1 = LinearGaussianCPD("x", [0.23], 0.56)
        cpd2 = LinearGaussianCPD("y", [0.67, 1, 4.56, 8], 2, ["x1", "x2", "x3"])
        self.assertEqual(cpd1.__str__(), "P(x) = N(0.23; 0.56)")
        self.assertEqual(
            cpd2.__str__(),
            "P(y | x1, x2, x3) = N(1.0*x1 + 4.56*x2 + 8.0*x3 + 0.67; 2)",
        )
