import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import OrientationF1, OrientationNPV, OrientationPrecision, OrientationRecall, OrientationSpecificity


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
    p.add_edges_from([("x1", "x2", "--"), ("x2", "x3", "--")])
    return p


def test_default_metrics(true_dag, est_dag):
    precision_result = OrientationPrecision().evaluate(true_dag, est_dag)
    recall_result = OrientationRecall().evaluate(true_dag, est_dag)
    f1_result = OrientationF1().evaluate(true_dag, est_dag)
    npv_result = OrientationNPV().evaluate(true_dag, est_dag)
    specificity_result = OrientationSpecificity().evaluate(true_dag, est_dag)

    for m in [precision_result, recall_result, f1_result, npv_result, specificity_result]:
        assert 0.0 <= m <= 1.0


def test_individual_scores(true_dag, est_dag):
    precision = OrientationPrecision().evaluate(true_dag, est_dag)
    recall = OrientationRecall().evaluate(true_dag, est_dag)
    f1 = OrientationF1().evaluate(true_dag, est_dag)
    npv = OrientationNPV().evaluate(true_dag, est_dag)
    specificity = OrientationSpecificity().evaluate(true_dag, est_dag)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)
    assert npv == pytest.approx(2 / 3)
    assert specificity == pytest.approx(2 / 3)


def test_perfect_match(true_dag):
    precision_result = OrientationPrecision().evaluate(true_dag, true_dag)
    recall_result = OrientationRecall().evaluate(true_dag, true_dag)
    f1_result = OrientationF1().evaluate(true_dag, true_dag)
    npv = OrientationNPV().evaluate(true_dag, true_dag)
    specificity = OrientationSpecificity().evaluate(true_dag, true_dag)
    assert precision_result == recall_result == f1_result == npv == specificity == 1.0


def test_empty_estimated_graphs(true_dag, empty_dag):
    est_empty_precision = OrientationPrecision().evaluate(true_dag, empty_dag)
    est_empty_recall = OrientationRecall().evaluate(true_dag, empty_dag)
    assert est_empty_precision == 0.0
    assert est_empty_recall == 0.0


def test_pdag_not_supported(pdag):
    """PDAGs are rejected; orientation requires directed graphs."""
    with pytest.raises(ValueError):
        OrientationPrecision().evaluate(pdag, pdag)
    with pytest.raises(ValueError):
        OrientationRecall().evaluate(pdag, pdag)
    with pytest.raises(ValueError):
        OrientationF1().evaluate(pdag, pdag)
    with pytest.raises(ValueError):
        OrientationNPV().evaluate(pdag, pdag)
    with pytest.raises(ValueError):
        OrientationSpecificity().evaluate(pdag, pdag)


def test_different_nodes_raises_recall(true_dag):
    with pytest.raises(ValueError):
        OrientationRecall().evaluate(true_dag, DAG([("x1", "x2")]))
    with pytest.raises(ValueError):
        OrientationPrecision().evaluate(true_dag, DAG(["x1", "x2"]))
    with pytest.raises(ValueError):
        OrientationF1().evaluate(true_dag, DAG(["x1", "x2"]))
    with pytest.raises(ValueError):
        OrientationNPV().evaluate(true_dag, DAG(["x1", "x2"]))
    with pytest.raises(ValueError):
        OrientationSpecificity().evaluate(true_dag, DAG(["x1", "x2"]))
