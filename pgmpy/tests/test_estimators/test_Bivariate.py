import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from pgmpy.estimators import CITests as ci_tests
from pgmpy.estimators.bivariate import ANM


class TestANM(unittest.TestCase):
    """Test suite for the ANM causal discovery function."""

    def setUp(self):
        """Set up common data and models for the tests."""
        self.sample_size = 200
        np.random.seed(42)

    def test_linear_gaussian_case(self):
        """
        Test ANM on a linear Gaussian model: Y = 2*X + N.
        ANM should not be able to distinguish the direction.
        """
        X = np.random.randn(self.sample_size)
        noise = np.random.randn(self.sample_size)
        Y = 2 * X + noise

        estimator = LinearRegression()
        # Pearson correlation is a suitable test for linear relationships
        ci_test = ci_tests.pearsonr

        direction = ANM(X, Y, regressor=estimator, independence_test=ci_test)
        # In the linear-Gaussian case, both directions fit well, so ANM is undecided.
        self.assertEqual(direction, "x--y")

    def test_nonlinear_case_wrong_estimator(self):
        """
        Test ANM on a non-linear model (Y = sin(X) + N) but with a linear estimator.
        The algorithm should fail to find the correct direction.
        """
        X = np.random.uniform(-5, 5, self.sample_size)
        noise = np.random.randn(self.sample_size) * 0.2
        Y = np.sin(X) + noise

        # A Linear Regression model is inappropriate for the sine function
        estimator = LinearRegression()
        ci_test = ci_tests.gcm

        direction = ANM(X, Y, regressor=estimator, independence_test=ci_test)
        # Because the estimator is wrong, the residuals won't be independent of the cause,
        # leading to an incorrect or undecided result.
        self.assertEqual(direction, "x--y")

    def test_disallowed_discrete_test(self):
        """
        Test that ANM raises a ValueError for disallowed discrete tests like chi_square.
        """
        X = np.random.randn(self.sample_size)
        Y = 2 * X + np.random.randn(self.sample_size)
        estimator = LinearRegression()

        # chi_square is a discrete test and should not be allowed.
        # The `with self.assertRaises(...)` block checks that a ValueError is correctly raised.
        with self.assertRaises(ValueError):
            ANM(X, Y, regressor=estimator, independence_test=ci_tests.chi_square)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
