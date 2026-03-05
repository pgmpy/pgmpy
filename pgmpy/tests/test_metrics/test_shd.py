import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SHD


@pytest.fixture
def shd_scorer():
    return SHD()


# Setup isolated node cases outside the decorator to avoid mutation across runs
isolated_3_a = DAG([(1, 2)])
isolated_3_a.add_node(3)
isolated_3_b = DAG([(1, 2)])
isolated_3_b.add_node(3)


# ---------------------------------------------------------------------------
# Standard SHD tests (DAG inputs)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CPDAG-aware SHD tests (PDAG inputs — automatically handled)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "graph1, graph2, expected_score",
    [
        # 1. Identical PDAG inputs score 0
        (PDAG([("X", "Z"), ("Y", "Z")], []), PDAG([("X", "Z"), ("Y", "Z")], []), 0),
        # 2. Uncompleted PDAGs are canonicalized via Meek's rules before scoring
        (PDAG([("X", "Y")], [("Y", "Z")]), PDAG([("X", "Y"), ("Y", "Z")], []), 0),
        # 3. Non-equivalent graphs accumulate penalties correctly
        (
            DAG([("X", "Y"), ("Y", "Z")]),
            PDAG([("X", "Z"), ("Y", "Z")], []),
            3,
        ),  # DAG vs non-equivalent PDAG
        # 4. Edge cases: isolated nodes score 0
        (isolated_3_a, isolated_3_b, 0),
    ],
)
def test_shd_pdag_scores(shd_scorer, graph1, graph2, expected_score):
    """SHD automatically uses CPDAG comparison when at least one input is a PDAG."""
    assert shd_scorer(graph1, graph2) == expected_score


def test_shd_pdag_is_symmetric(shd_scorer):
    """SHD(A, B) == SHD(B, A) for PDAG inputs."""
    dag = DAG([("X", "Y"), ("Y", "Z")])
    pdag = PDAG([("X", "Z"), ("Y", "Z")], [])
    assert shd_scorer(dag, pdag) == shd_scorer(pdag, dag)


def test_shd_pdag_unequal_nodes_raises(shd_scorer):
    """Different node sets raise ValueError."""
    p1 = PDAG([("X", "Y")], [])
    p2 = PDAG([("A", "B")], [])
    with pytest.raises(ValueError, match=r"The graphs must have the same nodes\."):
        shd_scorer(p1, p2)
