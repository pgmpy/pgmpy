import pytest

from pgmpy.base import DAG
from pgmpy.metrics import SHD


@pytest.fixture
def shd_scorer():
    return SHD()


def test_shd1(shd_scorer):
    dag1 = DAG([(1, 2)])
    dag2 = DAG([(2, 1)])
    assert shd_scorer(dag1, dag2) == 1


def test_shd2(shd_scorer):
    dag1 = DAG([(1, 2), (2, 4), (1, 3), (3, 4)])
    dag2 = DAG([(1, 2), (1, 3), (3, 2), (3, 4)])
    assert shd_scorer(dag1, dag2) == 2


def test_shd3(shd_scorer):
    dag1 = DAG([(1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)])
    dag2 = DAG([(1, 2), (1, 3), (4, 2), (3, 5), (4, 6), (5, 6)])
    assert shd_scorer(dag1, dag2) == 3


def test_shd_isolated_nodes(shd_scorer):
    dag1 = DAG([(1, 2)])
    dag1.add_nodes_from([3])
    dag2 = DAG([(1, 2), (2, 3)])

    assert shd_scorer(dag1, dag2) == 1
    assert shd_scorer(dag2, dag1) == 1


def test_shd_mixed_differences(shd_scorer):
    dag1 = DAG([(1, 2), (2, 3), (2, 4), (4, 5), (6, 5), (7, 8)])
    dag1.add_nodes_from([9, 10])
    dag2 = DAG([(1, 2), (2, 4), (5, 4), (6, 5), (8, 7), (9, 10)])
    dag2.add_nodes_from([3, 7])

    assert shd_scorer(dag1, dag2) == 4
    assert shd_scorer(dag2, dag1) == 4


def test_shd_unequal_graphs(shd_scorer):
    dag1 = DAG([(1, 2), (1, 3), (3, 2), (3, 4)])
    dag2 = DAG([(1, 2), (1, 3), (3, 2), (3, 5)])

    with pytest.raises(ValueError, match=r"The graphs must have the same nodes\."):
        shd_scorer(dag1, dag2)
