import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import VarSortability


@pytest.fixture
def recoverable_chain():
    """
    X -> Y -> Z with strong, well-separated coefficients so that SortnRegress
    recovers the chain exactly at threshold=0.3.
    """
    rng = np.random.default_rng(seed=42)
    n = 1000
    x = rng.normal(0, 0.1, n)
    y = 2.0 * x + rng.normal(0, 0.5, n)
    z = 2.0 * y + rng.normal(0, 0.5, n)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})
    dag = DAG([("X", "Y"), ("Y", "Z")])
    return data, dag


class Varsortabilityparams:
    def test_default_params(self):
        metric = VarSortability()
        assert metric.metric == "shd"
        assert metric.variant == "r2"
        assert metric.threshold == 0.3
        assert metric.estimator is None

    def test_params_stored_unmodified(self):
        metric = VarSortability(metric="shd", variant="varsortability", threshold=0.5)
        assert metric.variant == "varsortability"
        assert metric.threshold == 0.5

    def test_supported_graph_types(self):
        supported = VarSortability._tags["supported_graph_types"]
        assert DAG in supported
        assert PDAG in supported

    def test_tags_declare_no_true_graph_required(self):
        assert VarSortability._tags["requires_true_graph"] is False
        assert VarSortability._tags["requires_data"] is True


class TestVarsortabilityEvaluation:
    def test_returns_numeric_score(self, recoverable_chain):
        data, dag = recoverable_chain
        score = VarSortability().evaluate(X=data, causal_graph=dag)
        assert isinstance(score, (int, float, np.integer, np.floating))

    def test_zero_distance_on_recoverable_graph(self, recoverable_chain):
        """
        When SortnRegress recovers `causal_graph` exactly, the SHD between the
        two is 0.
        """
        data, dag = recoverable_chain
        assert VarSortability().evaluate(X=data, causal_graph=dag) == 0

    def test_mismatched_graph_scores(self, recoverable_chain):
        data, correct_dag = recoverable_chain
        wrong_dag = DAG([("Z", "X"), ("X", "Y")])

        metric = VarSortability()
        correct_score = metric.evaluate(X=data, causal_graph=correct_dag)
        wrong_score = metric.evaluate(X=data, causal_graph=wrong_dag)

        assert wrong_score > correct_score

    def test_invalid_variant(self, recoverable_chain):
        data, dag = recoverable_chain
        metric = VarSortability(variant="not_a_real_variant")
        with pytest.raises(ValueError, match="variant must be one of"):
            metric.evaluate(X=data, causal_graph=dag)


class TestVarsortabilityValidation:
    def test_reject_unsupervised_metric(self, recoverable_chain):
        data, dag = recoverable_chain
        metric = VarSortability(metric="varsortability")
        with pytest.raises(ValueError, match="not a supported supervised metric"):
            metric.evaluate(X=data, causal_graph=dag)

    def test_reject_non_dag_graph(self, recoverable_chain):
        data, _ = recoverable_chain
        with pytest.raises(ValueError):
            VarSortability().evaluate(X=data, causal_graph=pd.DataFrame())

    def test_rejects_zero_variance_column(self, recoverable_chain):
        data, dag = recoverable_chain
        data = data.copy()
        data["C"] = 0.0
        dag = DAG(list(dag.edges()))
        dag.add_node("C")

        with pytest.raises(ValueError, match="zero variance"):
            VarSortability().evaluate(X=data, causal_graph=dag)
