import pytest

from pgmpy.base import DAG
from pgmpy.metrics import SeparationDistance


@pytest.fixture
def parent_sd():
    return SeparationDistance(strategy="parent")


@pytest.fixture
def ancestor_sd():
    return SeparationDistance(strategy="ancestor")


@pytest.fixture
def parent_sd_sym():
    return SeparationDistance(strategy="parent", symmetric=True)


@pytest.fixture
def ancestor_sd_sym():
    return SeparationDistance(strategy="ancestor", symmetric=True)


def test_identical_graphs_parent(parent_sd):
    dag = DAG([("A", "B"), ("B", "C"), ("A", "D")])
    assert parent_sd(dag, dag) == 0.0


def test_identical_graphs_ancestor(ancestor_sd):
    dag = DAG([("A", "B"), ("B", "C"), ("A", "D")])
    assert ancestor_sd(dag, dag) == 0.0


def test_chain_vs_fork_parent(parent_sd):
    dag1 = DAG([("A", "B"), ("B", "C")])
    dag2 = DAG([("A", "B"), ("A", "C")])
    assert parent_sd(dag1, dag2) == pytest.approx(1 / 6)
    assert parent_sd(dag2, dag1) == pytest.approx(1 / 6)


def test_chain_vs_fork_ancestor(ancestor_sd):
    dag1 = DAG([("A", "B"), ("B", "C")])
    dag2 = DAG([("A", "B"), ("A", "C")])
    assert ancestor_sd(dag1, dag2) == pytest.approx(1 / 6)


def test_symmetric_parent(parent_sd_sym):
    dag1 = DAG([("A", "B"), ("B", "C")])
    dag2 = DAG([("A", "B"), ("A", "C")])
    assert parent_sd_sym(dag1, dag2) == pytest.approx(1 / 6)


def test_symmetric_identical(parent_sd_sym):
    dag = DAG([("A", "B"), ("B", "C"), ("A", "C")])
    assert parent_sd_sym(dag, dag) == 0.0


def test_markov_equivalent_parent(parent_sd_sym):
    dag1 = DAG([("A", "B"), ("B", "C")])
    dag2 = DAG([("C", "B"), ("B", "A")])
    assert parent_sd_sym(dag1, dag2) == pytest.approx(0.0)


def test_markov_equivalent_ancestor(ancestor_sd_sym):
    dag1 = DAG([("A", "B"), ("B", "C")])
    dag2 = DAG([("C", "B"), ("B", "A")])
    assert ancestor_sd_sym(dag1, dag2) == pytest.approx(0.0)


def test_collider_reversal(parent_sd):
    dag1 = DAG([("A", "B"), ("B", "C")])
    dag2 = DAG([("A", "B"), ("C", "B")])
    assert parent_sd(dag1, dag2) == pytest.approx(1 / 6)
    assert parent_sd(dag2, dag1) == pytest.approx(1 / 6)


def test_disconnected_vs_chain(parent_sd):
    dag_disc = DAG()
    dag_disc.add_nodes_from(["A", "B", "C"])
    dag_chain = DAG([("A", "B"), ("B", "C")])

    assert parent_sd(dag_chain, dag_disc) == pytest.approx(0.5)
    assert parent_sd(dag_disc, dag_chain) == pytest.approx(0.0)


def test_fully_connected_est(parent_sd):
    dag_full = DAG([("A", "B"), ("A", "C"), ("B", "C")])
    dag_sparse = DAG([("A", "B")])
    dag_sparse.add_node("C")

    assert parent_sd(dag_sparse, dag_full) == pytest.approx(0.0)


def test_isolated_nodes(parent_sd):
    dag1 = DAG([("A", "B")])
    dag1.add_node("C")
    dag2 = DAG([("A", "B"), ("B", "C")])
    assert parent_sd(dag2, dag1) == pytest.approx(2 / 6)


def test_larger_graph(parent_sd):
    dag1 = DAG([(1, 2), (2, 3), (3, 4), (4, 5)])
    dag2 = DAG([(1, 2), (2, 3), (2, 4), (4, 5)])
    result = parent_sd(dag1, dag2)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_single_node(parent_sd):
    dag = DAG()
    dag.add_node("A")
    assert parent_sd(dag, dag) == 0.0


def test_two_nodes_same_edge(parent_sd):
    dag = DAG([("A", "B")])
    assert parent_sd(dag, dag) == 0.0


def test_two_nodes_reversed_edge(parent_sd):
    dag1 = DAG([("A", "B")])
    dag2 = DAG([("B", "A")])
    assert parent_sd(dag1, dag2) == 0.0


def test_two_nodes_disconnected_vs_connected(parent_sd):
    dag_conn = DAG([("A", "B")])
    dag_disc = DAG()
    dag_disc.add_nodes_from(["A", "B"])

    assert parent_sd(dag_conn, dag_disc) == pytest.approx(0.5)
    assert parent_sd(dag_disc, dag_conn) == pytest.approx(0.0)


def test_paper_chain_vs_collider(parent_sd_sym):
    chain = DAG([("X1", "X2"), ("X2", "X3"), ("X3", "X4")])
    collider = DAG([("X1", "X2"), ("X2", "X3"), ("X4", "X3")])
    assert parent_sd_sym(chain, collider) > 0.0


def test_invalid_strategy():
    with pytest.raises(ValueError, match="strategy must be one of"):
        SeparationDistance(strategy="invalid")


def test_mismatched_nodes(parent_sd):
    dag1 = DAG([("A", "B")])
    dag2 = DAG([("C", "D")])
    with pytest.raises(ValueError, match="The graphs must have the same nodes"):
        parent_sd(dag1, dag2)


def test_non_dag_input(parent_sd):
    dag = DAG([("A", "B")])
    with pytest.raises(ValueError):
        parent_sd("not_a_dag", dag)
