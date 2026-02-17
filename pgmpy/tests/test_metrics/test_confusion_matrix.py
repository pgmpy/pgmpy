import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import ConfusionMatrix


@pytest.fixture
def true_dag():
    return DAG([('A', 'B'), ('B', 'C'), ('A', 'C')])


@pytest.fixture
def est_dag():
    return DAG([('A', 'B'), ('C', 'B')])


@pytest.fixture
def perfect_dag():
    return DAG([('A', 'B'), ('B', 'C'), ('A', 'C')])


@pytest.fixture
def empty_dag():
    dag = DAG()
    dag.add_nodes_from(['A', 'B', 'C'])
    return dag


def test_basic_adjacency_metrics(true_dag, est_dag):
    """Test basic adjacency metrics computation."""
    cm = ConfusionMatrix()
    result = cm.evaluate(true_dag, est_dag)

    assert 'adjacency_precision' in result
    assert 'adjacency_recall' in result
    assert 'adjacency_f1' in result
    assert 'adjacency_npv' in result
    assert 'adjacency_specificity' in result
    assert 'adjacency_confusion_matrix' in result

    adj_cm = result['adjacency_confusion_matrix']
    for key in ['tp', 'fp', 'fn', 'tn']:
        assert adj_cm[key] >= 0
        assert isinstance(adj_cm[key], int)

    for metric in ['adjacency_precision', 'adjacency_recall', 'adjacency_f1',
                   'adjacency_npv', 'adjacency_specificity']:
        assert 0.0 <= result[metric] <= 1.0


def test_perfect_match(true_dag, perfect_dag):
    """Test metrics when estimated graph perfectly matches true graph."""
    cm = ConfusionMatrix()
    result = cm.evaluate(true_dag, perfect_dag)

    assert result['adjacency_precision'] == 1.0
    assert result['adjacency_recall'] == 1.0
    assert result['adjacency_f1'] == 1.0

    adj_cm = result['adjacency_confusion_matrix']
    assert adj_cm['fp'] == 0
    assert adj_cm['fn'] == 0


def test_empty_graphs(empty_dag):
    """Test metrics with empty graphs."""
    cm = ConfusionMatrix()
    result = cm.evaluate(empty_dag, empty_dag)

    adj_cm = result['adjacency_confusion_matrix']
    assert adj_cm['tp'] == 0
    assert adj_cm['fp'] == 0
    assert adj_cm['fn'] == 0

    n_nodes = len(empty_dag.nodes())
    max_edges = n_nodes * (n_nodes - 1) // 2
    assert adj_cm['tn'] == max_edges


def test_orientation_metrics(true_dag, est_dag):
    """Test orientation metrics for DAGs."""
    cm = ConfusionMatrix()
    result = cm.evaluate(true_dag, est_dag)

    assert 'orientation_precision' in result
    assert 'orientation_recall' in result
    assert 'orientation_confusion_matrix' in result

    assert 0.0 <= result['orientation_precision'] <= 1.0
    assert 0.0 <= result['orientation_recall'] <= 1.0


def test_selective_metrics(true_dag, est_dag):
    """Test computation of selective metrics."""
    cm = ConfusionMatrix(metrics=['precision', 'recall'])
    result = cm.evaluate(true_dag, est_dag)

    assert 'adjacency_precision' in result
    assert 'adjacency_recall' in result
    assert 'adjacency_f1' not in result
    assert 'adjacency_npv' not in result
    assert 'adjacency_specificity' not in result


def test_pdag_support():
    """Test that PDAGs are supported."""
    pdag = PDAG()
    pdag.add_nodes_from(['A', 'B', 'C'])
    pdag.add_edges_from([('A', 'B'), ('B', 'C')])

    cm = ConfusionMatrix()
    result = cm.evaluate(pdag, pdag)

    assert 'adjacency_precision' in result
    assert 'adjacency_recall' in result
    assert 'orientation_precision' not in result


def test_different_nodes_error(true_dag):
    """Test error when graphs have different nodes."""
    other_dag = DAG([('X', 'Y')])
    cm = ConfusionMatrix()

    with pytest.raises(ValueError):
        cm.evaluate(true_dag, other_dag)


def test_edge_case_empty_estimated(true_dag, empty_dag):
    """Test edge case where estimated graph is empty."""
    cm = ConfusionMatrix()
    result = cm.evaluate(true_dag, empty_dag)

    assert result['adjacency_precision'] == 0.0
    assert result['adjacency_recall'] == 0.0


def test_confusion_matrix_values(true_dag, est_dag):
    """Test specific confusion matrix values for known graphs."""
    cm = ConfusionMatrix()
    result = cm.evaluate(true_dag, est_dag)

    adj_cm = result['adjacency_confusion_matrix']

    assert adj_cm['tp'] == 2
    assert adj_cm['fn'] == 1
    assert adj_cm['fp'] == 0
