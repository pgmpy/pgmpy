import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.metrics import VarSortability
from pgmpy.metrics.varsortability_score import compute_varsortability


@pytest.fixture
def increasing_variance_chain():
    """X -> Y -> Z with variance strictly increasing along the path."""
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame(
        {
            "X": np.random.normal(0, 0.1, n),
            "Y": np.random.normal(0, 1.0, n),
            "Z": np.random.normal(0, 10.0, n),
        }
    )
    dag = DAG([("X", "Y"), ("Y", "Z")])
    return data, dag


@pytest.fixture
def decreasing_variance_chain():
    """X -> Y -> Z with variance strictly decreasing along the path."""
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame(
        {
            "X": np.random.normal(0, 10.0, n),
            "Y": np.random.normal(0, 1.0, n),
            "Z": np.random.normal(0, 0.1, n),
        }
    )
    dag = DAG([("X", "Y"), ("Y", "Z")])
    return data, dag


class TestComputeVarsortability:
    def test_high_score_for_increasing_variance(self, increasing_variance_chain):
        data, dag = increasing_variance_chain
        score = compute_varsortability(data, dag)
        assert isinstance(score, float)
        assert 0.7 < score <= 1.0

    def test_low_score_for_decreasing_variance(self, decreasing_variance_chain):
        data, dag = decreasing_variance_chain
        score = compute_varsortability(data, dag)
        assert 0.0 <= score < 0.3

    def test_vacuously_true_for_empty_graph(self):
        # A graph with no edges has no causal paths to violate, so the score is 1.0.
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 3), columns=["X", "Y", "Z"])
        dag = DAG()
        dag.add_nodes_from(["X", "Y", "Z"])

        score = compute_varsortability(data, dag)
        assert score == 1.0

    def test_vacuously_true_for_single_node(self):
        np.random.seed(42)
        data = pd.DataFrame({"X": np.random.normal(0, 1, 100)})
        dag = DAG()
        dag.add_node("X")

        assert compute_varsortability(data, dag) == 1.0

    def test_near_equal_variance_weighted_as_half(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        data = pd.DataFrame({"X": x, "Y": x.copy()})
        dag = DAG([("X", "Y")])

        score = compute_varsortability(data, dag)
        assert score == pytest.approx(0.5)

    def test_tol_widens_near_equal_band(self):
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1.0, n)
        y = x * 1.0000001 + np.random.normal(0, 0.0001, n)
        data = pd.DataFrame({"X": x, "Y": y})
        dag = DAG([("X", "Y")])

        tight_score = compute_varsortability(data, dag, tol=1e-9)
        loose_score = compute_varsortability(data, dag, tol=1.0)

        assert loose_score == pytest.approx(0.5)
        assert tight_score != loose_score

    def test_diamond_graph_multiple_paths(self):
        np.random.seed(1)
        n = 2000
        x = np.random.normal(0, 0.5, n)
        y = x + np.random.normal(0, 1.0, n)
        z = x + np.random.normal(0, 1.0, n)
        w = y + z + np.random.normal(0, 5.0, n)
        data = pd.DataFrame({"X": x, "Y": y, "Z": z, "W": w})
        dag = DAG([("X", "Y"), ("X", "Z"), ("Y", "W"), ("Z", "W")])

        score = compute_varsortability(data, dag)
        assert score == 1.0

    def test_disconnected_node_does_not_affect_score(self):
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame(
            {
                "X": np.random.normal(0, 1, n),
                "Y": np.random.normal(0, 1, n),
                "Z": np.random.normal(0, 1, n) * 5,
            }
        )
        dag = DAG([("Y", "Z")])
        dag.add_node("X")

        score = compute_varsortability(data, dag)
        assert score == 1.0

    def test_missing_column_raises(self, increasing_variance_chain):
        data, dag = increasing_variance_chain
        incomplete_data = data.drop(columns=["Z"])

        with pytest.raises(KeyError):
            compute_varsortability(incomplete_data, dag)

    def test_score_bounded_between_zero_and_one(self, increasing_variance_chain, decreasing_variance_chain):
        for data, dag in (increasing_variance_chain, decreasing_variance_chain):
            score = compute_varsortability(data, dag)
            assert 0.0 <= score <= 1.0


class TestVarSortabilityClass:
    def test_default_tol(self):
        assert VarSortability().tol == 1e-9

    def test_custom_tol_stored(self):
        assert VarSortability(tol=0.05).tol == 0.05

    def test_supported_graph_types_includes_dag(self):
        assert DAG in VarSortability._tags["supported_graph_types"]

    def test_evaluate_matches_compute_varsortability(self, increasing_variance_chain):
        data, dag = increasing_variance_chain
        metric = VarSortability()

        assert metric.evaluate(X=data, causal_graph=dag) == compute_varsortability(data, dag)

    def test_evaluate_respects_tol(self):
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1.0, n)
        y = x * 1.0000001 + np.random.normal(0, 0.0001, n)
        data = pd.DataFrame({"X": x, "Y": y})
        dag = DAG([("X", "Y")])

        loose_metric = VarSortability(tol=1.0)
        assert loose_metric.evaluate(X=data, causal_graph=dag) == pytest.approx(0.5)

    def test_returns_float(self, increasing_variance_chain):
        data, dag = increasing_variance_chain
        result = VarSortability().evaluate(X=data, causal_graph=dag)
        assert isinstance(result, float)

    def test_rejects_unsupported_graph_type(self, increasing_variance_chain):
        data, _ = increasing_variance_chain
        with pytest.raises(ValueError):
            VarSortability().evaluate(X=data, causal_graph="not_a_dag")
