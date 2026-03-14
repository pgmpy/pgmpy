import pandas as pd
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import OrientationConfusionMatrix


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
def pdag():
    p = PDAG()
    p.add_nodes_from(["x1", "x2", "x3"])
    p.add_edges_from([("x1", "x2"), ("x2", "x3")])
    return p


def test_default_metrics(true_dag, est_dag):
    result = OrientationConfusionMatrix().evaluate(true_dag, est_dag)

    assert set(result) == {"cm", "precision", "recall", "f1", "npv", "specificity"}
    assert isinstance(result["cm"], pd.DataFrame)
    assert result["cm"].shape == (2, 2)
    for m in ["precision", "recall", "f1", "npv", "specificity"]:
        assert 0.0 <= result[m] <= 1.0


def test_selective_metrics(true_dag, est_dag):
    result = OrientationConfusionMatrix(metrics=["precision", "recall"]).evaluate(
        true_dag, est_dag
    )
    assert set(result) == {"precision", "recall"}


def test_confusion_matrix_values(true_dag, est_dag):
    """TP=4, FP=2, FN=2, TN=4; checks all cm entries and derived scalar metrics."""
    result = OrientationConfusionMatrix().evaluate(true_dag, est_dag)
    cm = result["cm"]

    assert cm.loc["Actual Present", "Est Present"] == 4  # TP
    assert cm.loc["Actual Absent", "Est Present"] == 2  # FP
    assert cm.loc["Actual Present", "Est Absent"] == 2  # FN
    assert cm.loc["Actual Absent", "Est Absent"] == 4  # TN

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["npv"] == pytest.approx(2 / 3)
    assert result["specificity"] == pytest.approx(2 / 3)


def test_perfect_match(true_dag):
    result = OrientationConfusionMatrix().evaluate(true_dag, true_dag)

    assert result["precision"] == result["recall"] == 1.0
    assert result["cm"].loc["Actual Absent", "Est Present"] == 0  # FP
    assert result["cm"].loc["Actual Present", "Est Absent"] == 0  # FN


def test_empty_estimated_graph(true_dag, empty_dag):
    result = OrientationConfusionMatrix().evaluate(true_dag, empty_dag)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["cm"].loc["Actual Present", "Est Present"] == 0  # TP
    assert result["cm"].loc["Actual Absent", "Est Present"] == 0  # FP
    assert result["cm"].loc["Actual Present", "Est Absent"] == 0  # FN


def test_pdag_not_supported(pdag):
    """PDAGs are rejected; orientation requires directed graphs."""
    with pytest.raises(ValueError):
        OrientationConfusionMatrix().evaluate(pdag, pdag)


def test_different_nodes_raises(true_dag):
    with pytest.raises(ValueError):
        OrientationConfusionMatrix().evaluate(true_dag, DAG([("x1", "x2")]))
