import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SHD


@pytest.fixture
def shd_scorer():
    return SHD()


@pytest.fixture
def shd_scorer_double():
    return SHD(edge_reverse_penalty=2)


def test_shd1(shd_scorer, shd_scorer_double):
    dag1 = DAG([(1, 2)])
    dag2 = DAG([(2, 1)])
    assert shd_scorer(dag1, dag2) == 1
    assert shd_scorer_double(dag1, dag2) == 2


def test_shd2(shd_scorer, shd_scorer_double):
    dag1 = DAG([(1, 2), (2, 4), (1, 3), (3, 4)])
    dag2 = DAG([(1, 2), (1, 3), (3, 2), (3, 4)])
    assert shd_scorer(dag1, dag2) == 2
    assert shd_scorer_double(dag1, dag2) == 2


def test_shd3(shd_scorer, shd_scorer_double):
    dag1 = DAG([(1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)])
    dag2 = DAG([(1, 2), (1, 3), (4, 2), (3, 5), (4, 6), (5, 6)])
    assert shd_scorer(dag1, dag2) == 3
    assert shd_scorer_double(dag1, dag2) == 4


def test_shd_isolated_nodes(shd_scorer, shd_scorer_double):
    dag1 = DAG([(1, 2)])
    dag1.add_nodes_from([3])
    dag2 = DAG([(1, 2), (2, 3)])

    assert shd_scorer(dag1, dag2) == 1
    assert shd_scorer(dag2, dag1) == 1
    assert shd_scorer_double(dag1, dag2) == 1
    assert shd_scorer_double(dag2, dag1) == 1


def test_shd_mixed_differences(shd_scorer, shd_scorer_double):
    dag1 = DAG([(1, 2), (2, 3), (2, 4), (4, 5), (6, 5), (7, 8)])
    dag1.add_nodes_from([9, 10])
    dag2 = DAG([(1, 2), (2, 4), (5, 4), (6, 5), (8, 7), (9, 10)])
    dag2.add_nodes_from([3, 7])

    assert shd_scorer(dag1, dag2) == 4
    assert shd_scorer(dag2, dag1) == 4
    assert shd_scorer_double(dag1, dag2) == 6
    assert shd_scorer_double(dag2, dag1) == 6


def test_shd_unequal_graphs(shd_scorer):
    dag1 = DAG([(1, 2), (1, 3), (3, 2), (3, 4)])
    dag2 = DAG([(1, 2), (1, 3), (3, 2), (3, 5)])

    with pytest.raises(ValueError):
        shd_scorer(dag1, dag2)


def test_shd_invalid_edge_reverse_penalty():
    with pytest.raises(ValueError, match="edge_reverse_penalty must be"):
        SHD(edge_reverse_penalty=3)


def test_shd_pdag_directed_vs_undirected_edge(shd_scorer, shd_scorer_double):
    pdag = PDAG(edge_list=[(1, 2, "->"), (2, 3, "--")])
    dag = DAG([(1, 2), (2, 3)])

    assert shd_scorer(pdag, dag) == 1
    assert shd_scorer(dag, pdag) == 1
    assert shd_scorer_double(pdag, dag) == 2
    assert shd_scorer_double(dag, pdag) == 2


def test_shd_pdag_undirected_edge_vs_missing_edge(shd_scorer, shd_scorer_double):
    pdag = PDAG(edge_list=[(1, 2, "--")])
    pdag.add_node(3)
    dag = DAG()
    dag.add_nodes_from([1, 2, 3])

    assert shd_scorer(pdag, dag) == 1
    assert shd_scorer(dag, pdag) == 1
    assert shd_scorer_double(pdag, dag) == 1
    assert shd_scorer_double(dag, pdag) == 1


def test_shd_pdag_reversed_directed_edge(shd_scorer, shd_scorer_double):
    pdag1 = PDAG(edge_list=[(1, 2, "->")])
    pdag2 = PDAG(edge_list=[(2, 1, "->")])

    assert shd_scorer(pdag1, pdag2) == 1
    assert shd_scorer(pdag2, pdag1) == 1
    assert shd_scorer_double(pdag1, pdag2) == 2
    assert shd_scorer_double(pdag2, pdag1) == 2
