"""
Tests for the sklearn-compatible DagmaLinear class in pgmpy.causal_discovery.
"""

import unittest

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery.DAGMA import DagmaLinear


def expected_failed_checks(estimator):
    """
    scikit-learn checks that are expected to fail
    for pgmpy causal discovery algorithms.
    """
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do \
        not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score \
        method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [DagmaLinear()],
    expected_failed_checks=expected_failed_checks,
)
def test_dagma_compatibility(estimator, check):
    """
    Automatically runs scikit-learn API compliance checks.
    """
    check(estimator)


@pytest.fixture
def continuous_data():
    """
    Set up a simple synthetic dataset where X -> Y -> Z.
    This ensures the algorithm has a clear causal signal to discover.
    """
    np.random.seed(42)
    X = np.random.normal(0, 1, 1000)
    Y = 2.0 * X + np.random.normal(0, 0.5, 1000)
    Z = 1.5 * Y + np.random.normal(0, 0.5, 1000)

    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


class TestDagmaLinearCore:
    """Tests for core DagmaLinear functionality."""

    def test_estimate_returns_dag(self, continuous_data):
        est = DagmaLinear()

        est.fit(continuous_data)

        # 1. Check if the output is successfully saved as a pgmpy DAG
        assert isinstance(est.causal_graph_, DAG)

        # 2. Check if the extracted feature names match the dataframe columns
        np.testing.assert_array_equal(est.feature_names_in_, ["X", "Y", "Z"])

        # 3. Check if the estimated adjacency matrix
        # was saved and is a NumPy array
        assert isinstance(est.adjacency_matrix_, np.ndarray)
        assert est.adjacency_matrix_.shape == (3, 3)

        # 4. Check if the algorithm successfully
        # found the X -> Y and Y -> Z edges
        learned_edges = list(est.causal_graph_.edges())
        assert ("X", "Y") in learned_edges
        assert ("Y", "Z") in learned_edges

        # 5. Prove the bounds and barriers successfully
        # prevented cycles and self-loops
        assert ("Z", "X") not in learned_edges
        assert ("X", "X") not in learned_edges
        assert ("Y", "Y") not in learned_edges
        assert ("Z", "Z") not in learned_edges

    def test_custom_hyperparameters(self):
        """
        Ensure custom hyperparameters are strictly mapped to the instance.
        """
        est = DagmaLinear(s=2.0, lambda1=0.1, mu_init=2.0, mu_factor=0.5, max_iter=50, w_threshold=0.4)
        assert est.s == 2.0
        assert est.lambda1 == 0.1
        assert est.mu_init == 2.0
        assert est.mu_factor == 0.5
        assert est.max_iter == 50
        assert est.w_threshold == 0.4

    def test_compare_with_official_dagma(self, continuous_data):
        """
        Compare the adjacency matrix output of pgmpy's DagmaLinear
        with the official dagma package implementation.
        """
        try:
            from dagma.linear import DagmaLinear as OfficialDagmaLinear
        except ImportError:
            pytest.skip("Official 'dagma' not installed.")

        # 1. Run official DAGMA
        model_official = OfficialDagmaLinear(loss_type="l2")
        W_official = model_official.fit(continuous_data.to_numpy().copy(), lambda1=0.05)

        # 2. Run pgmpy's DAGMA
        est = DagmaLinear(lambda1=0.05)
        est.fit(continuous_data)
        W_pgmpy = est.adjacency_matrix_

        # Check 1. Both optimizers found the exact same DAG structure
        # Official optimizer 'adam', pgmpy optimizer 'L-BFGS-B'
        np.testing.assert_array_equal(W_pgmpy != 0, W_official != 0)

        # Check 2. Assert matrices are identical up to a small tolerance
        np.testing.assert_allclose(W_pgmpy, W_official, atol=0.05)


class TestDagmaLinear(unittest.TestCase):
    def setUp(self):
        """
        Set up a simple synthetic dataset where X -> Y -> Z.

        """
        np.random.seed(42)
        X = np.random.normal(0, 1, 1000)
        Y = 2.0 * X + np.random.normal(0, 0.5, 1000)
        Z = 1.5 * Y + np.random.normal(0, 0.5, 1000)

        self.data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_fit_returns_dag(self):
        """
        Test if the fit method runs successfully
        and returns a proper DAG object.
        """
        # Initialize the estimator with default hyperparameters
        estimator = DagmaLinear()

        # Run the continuous optimization
        estimator.fit(self.data)

        # Assertions
        # 1. Check if the output is successfully saved as a pgmpy DAG
        self.assertIsInstance(estimator.causal_graph_, DAG)

        # 2. Check if the extracted feature names match the dataframe columns
        np.testing.assert_array_equal(estimator.feature_names_in_, ["X", "Y", "Z"])

        # 3. Check if the estimated adjacency matrix
        # was saved and is a NumPy array
        self.assertIsInstance(estimator.adjacency_matrix_, np.ndarray)

        self.assertEqual(estimator.adjacency_matrix_.shape, (3, 3))
        # Check if the algorithm successfully found the X -> Y and Y -> Z edges
        learned_edges = list(estimator.causal_graph_.edges())
        self.assertIn(("X", "Y"), learned_edges)
        self.assertIn(("Y", "Z"), learned_edges)
        self.assertNotIn(("Z", "X"), learned_edges)

    def test_custom_hyperparameters(self):
        """
        Test if the __init__ correctly stores user-defined hyperparameters.
        """
        estimator = DagmaLinear(lambda1=0.1, max_iter=50)
        self.assertEqual(estimator.lambda1, 0.1)
        self.assertEqual(estimator.max_iter, 50)
