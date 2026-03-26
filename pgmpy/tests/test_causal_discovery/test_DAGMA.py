"""
Tests for the sklearn-compatible DagmaLinear class in pgmpy.causal_discovery.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
import unittest
from pgmpy.base import DAG
from pgmpy.causal_discovery.DAGMA import DagmaLinear
from pgmpy.causal_discovery import ExpertKnowledge


def expected_failed_checks(estimator):
    """
    scikit-learn checks that are expected to fail
    for pgmpy causal discovery algorithms.
    """
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
        "check_estimators_dtypes": "DagmaLinear currently only supports continuous numerical data.",
        "check_estimators_fit_returns_self": "scikit-learn check requires fit(X, y) but we only use X.",
        "check_pipeline_consistency": "Causal discovery estimators do not support predict methods.",
        "check_fit2d_1feature": "Causal discovery requires at least 2 features."
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
        est = DagmaLinear(
            s=2.0,
            lambda1=0.1,
            mu_init=2.0,
            mu_factor=0.5,
            max_iter=50,
            w_threshold=0.4
        )
        assert est.s == 2.0
        assert est.lambda1 == 0.1
        assert est.mu_init == 2.0
        assert est.mu_factor == 0.5
        assert est.max_iter == 50
        assert est.w_threshold == 0.4


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
        np.testing.assert_array_equal(estimator.feature_names_in_,
                                      ["X", "Y", "Z"])

        # 3. Check if the estimated adjacency matrix
        # was saved and is a NumPy array
        self.assertIsInstance(estimator.adjacency_matrix_, np.ndarray)

        self.assertEqual(estimator.adjacency_matrix_.shape, (3, 3))
        # Check if the algorithm successfully found the X -> Y and Y -> Z edges
        learned_edges = list(estimator.causal_graph_.edges())
        self.assertIn(("X", "Y"), learned_edges)
        self.assertIn(("Y", "Z"), learned_edges)
        self.assertNotIn(("Z", "X"),
                         learned_edges)  # Prove it didn't draw a cycle

    def test_custom_hyperparameters(self):
        """
        Test if the __init__ correctly stores user-defined hyperparameters.
        """
        estimator = DagmaLinear(lambda1=0.1, max_iter=50)
        self.assertEqual(estimator.lambda1, 0.1)
        self.assertEqual(estimator.max_iter, 50)


class TestDagmaExpertKnowledge:
    """Tests for DagmaLinear with expert knowledge constraints."""

    def test_forbidden_edges(self, continuous_data):
        """
        Test if the algorithm strictly obeys forbidden edges.
        The natural signal in continuous_data is X -> Y -> Z.
        We will forbid X -> Y and prove it does not appear in the final graph.
        """
        # Forbid the true causal edge
        expert_knowledge = ExpertKnowledge(forbidden_edges=[("X", "Y")])

        # Pass expert_knowledge to the constructor, NOT to fit()
        est = DagmaLinear(expert_knowledge=expert_knowledge)
        est.fit(continuous_data)

        # Assert that the forbidden edge was successfully masked out
        learned_edges = list(est.causal_graph_.edges())
        assert ("X", "Y") not in learned_edges

    def test_temporal_order_excludes_backward_edges(self, continuous_data):
        """
        Test if the algorithm translates temporal tiers into forbidden edges
        and restricts edges that go backward in time.
        """
        # Define a timeline where Z happens first, then Y, then X
        # (This is completely backward from the true data generation process)
        expert_knowledge = ExpertKnowledge(temporal_order=[["Z"],
                                                           ["Y"],
                                                           ["X"]])

        # Pass expert_knowledge to the constructor, NOT to fit()
        est = DagmaLinear(expert_knowledge=expert_knowledge)
        est.fit(continuous_data)

        learned_edges = list(est.causal_graph_.edges())

        # Because Z comes before Y and Y before X,
        # X->Y and Y->Z are now "backward in time"
        # The algorithm must not draw them.
        assert ("X", "Y") not in learned_edges
        assert ("Y", "Z") not in learned_edges

        # Verify that ANY edges it did draw strictly obey the temporal ordering
        temporal_ordering = expert_knowledge.temporal_ordering
        for u, v in learned_edges:
            assert temporal_ordering[u] <= temporal_ordering[v]
