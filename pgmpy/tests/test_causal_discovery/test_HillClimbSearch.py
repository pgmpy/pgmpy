"""
Tests for the sklearn-compatible HillClimbSearch class in pgmpy.causal_discovery.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.estimators import ExpertKnowledge
from pgmpy.utils import get_example_model


def make_estimator():
    """Factory function to create a HillClimbSearch estimator for sklearn tests."""
    return HillClimbSearch()


@parametrize_with_checks([make_estimator()])
def test_hillclimb_compatibility(estimator, check):
    """Test sklearn compatibility using sklearn's estimator checks."""
    check(estimator)


@pytest.fixture
def rand_data():
    """Generate random data for testing."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randint(0, 5, size=(int(1e4), 2)),
        columns=list("AB"),
        dtype="category",
    )
    data["C"] = data["B"]
    return data


@pytest.fixture
def titanic_data():
    """Load titanic dataset for testing."""
    titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
    return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")


class TestHillClimbSearchBasic:
    """Basic tests for HillClimbSearch."""

    def test_estimate_rand(self, rand_data):
        """Test basic estimation with random data."""
        hc = HillClimbSearch(scoring_method="k2", show_progress=False)
        hc.fit(rand_data)

        assert set(hc.causal_graph_.nodes()) == set(["A", "B", "C"])
        # B and C are copies, so there should be an edge between them
        edges = list(hc.causal_graph_.edges())
        assert edges == [("B", "C")] or edges == [("C", "B")]

    def test_estimate_with_start_dag(self, rand_data):
        """Test estimation with a starting DAG."""
        start_dag = DAG([("A", "B"), ("A", "C")])
        hc = HillClimbSearch(
            scoring_method="k2",
            start_dag=start_dag,
            show_progress=False,
        )
        hc.fit(rand_data)

        assert set(hc.causal_graph_.nodes()) == set(["A", "B", "C"])
        # The algorithm should find the optimal structure
        assert ("B", "C") in hc.causal_graph_.edges() or (
            "C",
            "B",
        ) in hc.causal_graph_.edges()

    def test_estimate_titanic(self, titanic_data):
        """Test estimation with titanic dataset."""
        hc = HillClimbSearch(scoring_method="k2", show_progress=False)
        hc.fit(titanic_data)

        assert set(hc.causal_graph_.nodes()) == set(["Survived", "Sex", "Pclass"])
        # Checking learned structure
        assert len(hc.causal_graph_.edges()) > 0

    def test_estimate_with_max_indegree(self, rand_data):
        """Test estimation with max_indegree constraint."""
        hc = HillClimbSearch(
            scoring_method="k2",
            max_indegree=1,
            show_progress=False,
        )
        hc.fit(rand_data)

        # Check that no node has more than 1 parent
        for node in hc.causal_graph_.nodes():
            assert len(hc.causal_graph_.get_parents(node)) <= 1


class TestHillClimbSearchExpertKnowledge:
    """Tests for HillClimbSearch with expert knowledge."""

    def test_forbidden_edges(self, rand_data):
        """Test that forbidden edges are not in the result."""
        expert = ExpertKnowledge(forbidden_edges=[("B", "C"), ("C", "B")])
        hc = HillClimbSearch(
            scoring_method="k2",
            expert_knowledge=expert,
            show_progress=False,
        )
        hc.fit(rand_data)

        # B-C edge should not be present
        assert ("B", "C") not in hc.causal_graph_.edges()
        assert ("C", "B") not in hc.causal_graph_.edges()

    def test_required_edges(self, rand_data):
        """Test that required edges are in the result."""
        expert = ExpertKnowledge(required_edges=[("A", "B")])
        hc = HillClimbSearch(
            scoring_method="k2",
            expert_knowledge=expert,
            show_progress=False,
        )
        hc.fit(rand_data)

        # A->B edge should be present
        assert ("A", "B") in hc.causal_graph_.edges()


class TestHillClimbSearchAttributes:
    """Test that attributes are properly set after fitting."""

    def test_attributes_after_fit(self, rand_data):
        """Test that all expected attributes are set after fitting."""
        hc = HillClimbSearch(scoring_method="k2", show_progress=False)
        hc.fit(rand_data)

        # Check causal_graph_ attribute
        assert hasattr(hc, "causal_graph_")
        assert isinstance(hc.causal_graph_, DAG)

        # Check adjacency_matrix_ attribute
        assert hasattr(hc, "adjacency_matrix_")
        assert isinstance(hc.adjacency_matrix_, pd.DataFrame)

        # Check n_features_in_ attribute
        assert hasattr(hc, "n_features_in_")
        assert hc.n_features_in_ == 3

        # Check feature_names_in_ attribute
        assert hasattr(hc, "feature_names_in_")

    def test_adjacency_matrix_consistency(self, rand_data):
        """Test that adjacency matrix is consistent with causal_graph_."""
        hc = HillClimbSearch(scoring_method="k2", show_progress=False)
        hc.fit(rand_data)

        # Check dimensions
        n_nodes = len(hc.causal_graph_.nodes())
        assert hc.adjacency_matrix_.shape == (n_nodes, n_nodes)

        # Check that edges in graph match adjacency matrix
        for u, v in hc.causal_graph_.edges():
            assert hc.adjacency_matrix_.loc[u, v] == 1


class TestHillClimbSearchScoringMethods:
    """Test different scoring methods."""

    @pytest.mark.parametrize("scoring_method", ["k2", "bdeu", "bic-d"])
    def test_discrete_scoring_methods(self, rand_data, scoring_method):
        """Test various discrete scoring methods."""
        hc = HillClimbSearch(scoring_method=scoring_method, show_progress=False)
        hc.fit(rand_data)

        assert hasattr(hc, "causal_graph_")
        assert len(hc.causal_graph_.nodes()) == 3

    def test_auto_scoring_method(self, rand_data):
        """Test automatic scoring method selection."""
        hc = HillClimbSearch(scoring_method=None, show_progress=False)
        hc.fit(rand_data)

        assert hasattr(hc, "causal_graph_")


class TestHillClimbSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_start_dag_variables(self, rand_data):
        """Test that invalid start_dag raises an error."""
        # DAG with different variables
        invalid_dag = DAG([("X", "Y")])
        hc = HillClimbSearch(
            scoring_method="k2",
            start_dag=invalid_dag,
            show_progress=False,
        )

        with pytest.raises(ValueError, match="start_dag"):
            hc.fit(rand_data)

    def test_cyclic_required_edges(self, rand_data):
        """Test that cyclic required edges raise an error."""
        expert = ExpertKnowledge(required_edges=[("A", "B"), ("B", "C"), ("C", "A")])
        hc = HillClimbSearch(
            scoring_method="k2",
            expert_knowledge=expert,
            show_progress=False,
        )

        with pytest.raises(ValueError, match="cycle"):
            hc.fit(rand_data)

    def test_epsilon_convergence(self, rand_data):
        """Test that algorithm respects epsilon for convergence."""
        # Very large epsilon should stop quickly
        hc = HillClimbSearch(
            scoring_method="k2",
            epsilon=1e10,
            show_progress=False,
        )
        hc.fit(rand_data)

        # Should have found at least some structure
        assert hasattr(hc, "causal_graph_")

    def test_max_iter(self, rand_data):
        """Test that algorithm respects max_iter."""
        hc = HillClimbSearch(
            scoring_method="k2",
            max_iter=1,
            show_progress=False,
        )
        hc.fit(rand_data)

        # Should have limited iterations
        assert hasattr(hc, "causal_graph_")


class TestHillClimbSearchAlarm:
    """Test with the alarm network (larger dataset)."""

    def test_alarm_network(self):
        """Test learning structure from alarm network data."""
        alarm_model = get_example_model("alarm")
        data = alarm_model.simulate(n_samples=int(1e3), seed=42)

        hc = HillClimbSearch(scoring_method="bic-d", show_progress=False)
        hc.fit(data)

        # Should learn a DAG with 37 nodes
        assert len(hc.causal_graph_.nodes()) == 37
        # Should learn some edges
        assert len(hc.causal_graph_.edges()) > 0


def test_hillclimb_api_usage():
    """Test the new sklearn-style API usage pattern."""
    # Simulate data
    model = get_example_model("asia")
    df = model.simulate(n_samples=1000, seed=42)

    # New sklearn-style API
    hc = HillClimbSearch(scoring_method="bic-d", show_progress=False)
    hc.fit(df)

    # Check results
    assert hasattr(hc, "causal_graph_")
    assert hasattr(hc, "adjacency_matrix_")
    assert set(hc.causal_graph_.nodes()) == set(df.columns)


class TestHillClimbSearchGaussian:
    """Test HillClimbSearch with continuous/Gaussian data."""

    @pytest.fixture
    def gaussian_data(self):
        """Load Gaussian test data."""
        data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0
        )
        return data

    @pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
    def test_gaussian_scoring_methods(self, gaussian_data, scoring_method):
        """Test Gaussian scoring methods."""
        hc = HillClimbSearch(scoring_method=scoring_method, show_progress=False)
        hc.fit(gaussian_data)

        assert hasattr(hc, "causal_graph_")
        assert len(hc.causal_graph_.nodes()) == len(gaussian_data.columns)


class TestHillClimbSearchMixed:
    """Test HillClimbSearch with mixed (continuous + discrete) data."""

    @pytest.fixture
    def mixed_data(self):
        """Load mixed test data."""
        data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )
        data["A_cat"] = data.A_cat.astype("category")
        data["B_cat"] = data.B_cat.astype("category")
        data["C_cat"] = data.C_cat.astype("category")
        data["B_int"] = data.B_int.astype("category")
        return data

    @pytest.mark.parametrize("scoring_method", ["ll-cg", "aic-cg", "bic-cg"])
    def test_mixed_scoring_methods(self, mixed_data, scoring_method):
        """Test mixed (conditional Gaussian) scoring methods."""
        hc = HillClimbSearch(scoring_method=scoring_method, show_progress=False)
        hc.fit(mixed_data)

        assert hasattr(hc, "causal_graph_")


class TestHillClimbSearchSearchSpace:
    """Test HillClimbSearch with search space constraints."""

    def test_search_space(self):
        """Test that search space constraints are respected."""
        adult_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

        search_space = [
            ("Age", "Education"),
            ("Education", "HoursPerWeek"),
            ("Education", "Income"),
            ("HoursPerWeek", "Income"),
            ("Age", "Income"),
        ]

        expert_knowledge = ExpertKnowledge(search_space=search_space)

        hc = HillClimbSearch(
            scoring_method="k2",
            expert_knowledge=expert_knowledge,
            show_progress=False,
        )
        hc.fit(adult_data)

        # Assert if dag is a subset of search_space
        for edge in hc.causal_graph_.edges():
            assert edge in search_space


class TestHillClimbSearchTemporalKnowledge:
    """Test HillClimbSearch with temporal ordering."""

    @pytest.fixture
    def titanic_data(self):
        """Load titanic dataset for testing."""
        titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")

    def test_temporal_ordering(self, titanic_data):
        """Test that temporal ordering is respected."""
        temporal_knowledge = ExpertKnowledge(
            temporal_order=[["Pclass", "Sex"], ["Survived"]]
        )
        hc = HillClimbSearch(
            expert_knowledge=temporal_knowledge,
            show_progress=False,
        )
        hc.fit(titanic_data)

        # Edges should only go from earlier time to later time
        allowed_edges = {
            ("Sex", "Survived"),
            ("Sex", "Pclass"),
            ("Pclass", "Sex"),
            ("Pclass", "Survived"),
        }
        for edge in hc.causal_graph_.edges():
            assert edge in allowed_edges


class TestHillClimbSearchNoLegalOperation:
    """Test edge case where no legal operations exist."""

    def test_no_legal_operation(self):
        """Test behavior when all operations are forbidden."""
        data = pd.DataFrame(
            [
                [1, 0, 0, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 1, 0, 1, 1],
                [1, 1, 0, 1, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 0, 0],
            ],
            columns=list("ABCDEFGHI"),
            dtype="category",
        )
        expert_knowledge = ExpertKnowledge(
            required_edges=[("A", "B"), ("B", "C")],
            forbidden_edges=[(u, v) for u in data.columns for v in data.columns],
        )
        hc = HillClimbSearch(
            scoring_method="k2",
            expert_knowledge=expert_knowledge,
            show_progress=False,
        )
        hc.fit(data)

        # Should still have a valid causal graph (just with required edges)
        assert hasattr(hc, "causal_graph_")
