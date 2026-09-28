import pandas as pd
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyPrecision, AdjacencyRecall, AdjacencyF1


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
    pdag.add_edges_from([("x1", "x2", "--"), ("x2", "x3", "--"), ("x3", "x4", "--"), ("x4", "x5", "--")])
    return pdag


@pytest.fixture
def est_pdag():
    # skeleton: x1-x2, x2-x3, x1-x3  (3 edges)
    pdag = PDAG()
    pdag.add_nodes_from(["x1", "x2", "x3", "x4", "x5"])
    pdag.add_edges_from([("x1", "x2", "--"), ("x2", "x3", "--"), ("x1", "x3", "--")])
    return pdag


def test_default_metrics(true_dag, est_dag):
    precision_result = AdjacencyPrecision().evaluate(true_dag, est_dag)
    recall_result = AdjacencyRecall().evaluate(true_dag,est_dag)
    f1_result = AdjacencyF1().evaluate(true_dag, est_dag)
    # npv_result =
    # specificity_result =

    for m in [precision_result, recall_result, f1_result]:
        assert 0.0 <= m <= 1.0

def test_precision(true_dag, est_dag):
    assert precision == pytest.approx(6 / 7)

def test_recall(true_dag, est_dag):
    assert recall = pytest.approx(6 / 8)

def test_f1_score(true_dag, est_dag):
    assert f1 == pytest.approx(4 / 5)

def test_npv(true_dag, est_dag):
    # assert result["npv"] == pytest.approx(1 / 3)
    pass

def test_specificty(true_dag, est_dag):
    # assert result["specificity"] == pytest.approx(1 / 2)
    pass


def test_perfect_match(true_dag):
    precision_result = AdjacencyPrecision().evaluate(true_dag, true_dag)
    recall_result = AdjacencyRecall().evaluate(true_dag,true_dag)
    f1_result = AdjacencyF1().evaluate(true_dag, true_dag)

    assert precision_result == recall_result == f1_result == 1.0


def test_empty_graphs(true_dag, empty_dag):
    true_empty_precision = AdjacencyPrecision(empty_dag, empty_dag)
    true_empty_recall = AdjacencyRecall(empty_dag, empty_dag)
    assert true_empty_precision == 0.0
    assert true_empty_recall == 0.0

    est_empty_precision = AdjacencyPrecision(true_dag, empty_dag)
    est_empty_recall = AdjacencyRecall(true_dag, empty_dag)
    assert est_empty_precision == 0.0
    assert est_empty_recall == 0.0


def test_pdag_support_precision(true_pdag, est_pdag):
    # TP=2 (x1-x2, x2-x3), FP=1 (x1-x3), FN=2 (x3-x4, x4-x5), TN=5
    result = AdjacencyPrecision().evaluate(true_pdag, est_pdag)
    assert result == pytest.approx(2 / 3)

def test_pdag_support_recall(true_pdag, est_pdag):
    result = AdjacencyRecall().evaluate(true_pdag, est_pdag)
    assert result == pytest.approx(1 / 2)

def test_pdag_support_f1(true_pdag, est_pdag):
    result = AdjacencyF1().evaluate(true_pdag, est_pdag)
    assert result == pytest.approx(4 / 7)
    assert result["npv"] == pytest.approx(5 / 7)
    assert result["specificity"] == pytest.approx(5 / 6)


def test_different_nodes_raises_recall(true_dag):
    with pytest.raises(ValueError):
        AdjacencyRecall().evaluate(true_dag, DAG([("x1", "x2")]))

def test_different_nodes_raises_precision(true_dag):
    with pytest.raises(ValueError):
        AdjacencyPrecision.evaluate(true_dag, DAG(["x1", "x2"]))

def test_different_nodes_raises_f1(true_dag):
    with pytest.raises(ValueError):
        AdjacencyF1.evaluate(true_dag, DAG(["x1", "x2"]))
