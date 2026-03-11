import pandas as pd
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyConfusionMatrix


# The models in true_dag and est_dag fixtures are taken from the paper: https://arxiv.org/pdf/2412.10039
@pytest.fixture
def true_dag():
    return DAG(
        [
            ("x1", "x2"),
            ("x1", "x4"),
            ("x1", "x5"),
            ("x2", "x5"),
            ("x2", "x3"),
            ("x2", "x4"),
            ("x4", "x5"),
            ("x5", "x3"),
        ]
    )


@pytest.fixture
def est_dag():
    return DAG(
        [
            ("x1", "x2"),
            ("x1", "x3"),
            ("x1", "x4"),
            ("x1", "x5"),
            ("x3", "x2"),
            ("x4", "x2"),
            ("x5", "x3"),
        ]
    )


@pytest.fixture
def empty_dag():
    dag = DAG()
    dag.add_nodes_from(["x1", "x2", "x3", "x4", "x5"])
    return dag


@pytest.fixture
def true_pdag():
    # skeleton: x1-x2, x2-x3, x3-x4, x4-x5  (4 edges, 6 non-edges over 5 nodes)
    pdag = PDAG()
    pdag.add_nodes_from(["x1", "x2", "x3", "x4", "x5"])
    pdag.add_edges_from([("x1", "x2"), ("x2", "x3"), ("x3", "x4"), ("x4", "x5")])
    return pdag


@pytest.fixture
def est_pdag():
    # skeleton: x1-x2, x2-x3, x1-x3  (3 edges)
    pdag = PDAG()
    pdag.add_nodes_from(["x1", "x2", "x3", "x4", "x5"])
    pdag.add_edges_from([("x1", "x2"), ("x2", "x3"), ("x1", "x3")])
    return pdag


def test_default_metrics(true_dag, est_dag):
    result = AdjacencyConfusionMatrix().evaluate(true_dag, est_dag)

    assert set(result) == {"cm", "precision", "recall", "f1", "npv", "specificity"}
    assert isinstance(result["cm"], pd.DataFrame)
    assert result["cm"].shape == (2, 2)
    for m in ["precision", "recall", "f1", "npv", "specificity"]:
        assert 0.0 <= result[m] <= 1.0


def test_selective_metrics(true_dag, est_dag):
    """Only requested metrics are returned."""
    result = AdjacencyConfusionMatrix(metrics=["precision", "recall"]).evaluate(
        true_dag, est_dag
    )
    assert set(result) == {"precision", "recall"}


def test_confusion_matrix_values(true_dag, est_dag):
    """TP=6, FP=1, FN=2, TN=1; checks all cm entries and derived scalar metrics."""
    result = AdjacencyConfusionMatrix().evaluate(true_dag, est_dag)
    cm = result["cm"]

    assert cm.loc["Actual Present", "Est Present"] == 6  # TP
    assert cm.loc["Actual Absent", "Est Present"] == 1  # FP
    assert cm.loc["Actual Present", "Est Absent"] == 2  # FN
    assert cm.loc["Actual Absent", "Est Absent"] == 1  # TN

    assert result["precision"] == pytest.approx(6 / 7)
    assert result["recall"] == pytest.approx(6 / 8)
    assert result["f1"] == pytest.approx(4 / 5)
    assert result["npv"] == pytest.approx(1 / 3)
    assert result["specificity"] == pytest.approx(1 / 2)


def test_perfect_match(true_dag):
    result = AdjacencyConfusionMatrix().evaluate(true_dag, true_dag)

    assert result["precision"] == result["recall"] == result["f1"] == 1.0
    assert result["cm"].loc["Actual Absent", "Est Present"] == 0  # FP
    assert result["cm"].loc["Actual Present", "Est Absent"] == 0  # FN


def test_empty_graphs(true_dag, empty_dag):
    both_empty = AdjacencyConfusionMatrix().evaluate(empty_dag, empty_dag)
    assert both_empty["cm"].loc["Actual Absent", "Est Absent"] == 10  # TN = C(5,2)

    est_empty = AdjacencyConfusionMatrix().evaluate(true_dag, empty_dag)
    assert est_empty["precision"] == 0.0
    assert est_empty["recall"] == 0.0


def test_pdag_support(true_pdag, est_pdag):
    # TP=2 (x1-x2, x2-x3), FP=1 (x1-x3), FN=2 (x3-x4, x4-x5), TN=5
    result = AdjacencyConfusionMatrix().evaluate(true_pdag, est_pdag)
    cm = result["cm"]

    assert cm.loc["Actual Present", "Est Present"] == 2  # TP
    assert cm.loc["Actual Absent", "Est Present"] == 1  # FP
    assert cm.loc["Actual Present", "Est Absent"] == 2  # FN
    assert cm.loc["Actual Absent", "Est Absent"] == 5  # TN

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["f1"] == pytest.approx(4 / 7)
    assert result["npv"] == pytest.approx(5 / 7)
    assert result["specificity"] == pytest.approx(5 / 6)


def test_different_nodes_raises(true_dag):
    with pytest.raises(ValueError):
        AdjacencyConfusionMatrix().evaluate(true_dag, DAG([("x1", "x2")]))
