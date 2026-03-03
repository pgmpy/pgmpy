import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import CPDAGSHD


@pytest.fixture
def cpdag_scorer():
    return CPDAGSHD()


# Setup isolated node cases safely outside the decorator
missing_y_edge = DAG([("X", "Z")])
missing_y_edge.add_node("Y")

isolated_3_a = DAG([(1, 2)])
isolated_3_a.add_node(3)
isolated_3_b = DAG([(1, 2)])
isolated_3_b.add_node(3)


@pytest.mark.parametrize(
    "graph1, graph2, expected_score",
    [
        # 1. Equivalent DAGs map to the same score (0)
        (
            DAG([("X", "Y"), ("Y", "Z")]),
            DAG([("Y", "X"), ("Y", "Z")]),
            0,
        ),  # Chain vs Fork
        (DAG([(1, 2)]), DAG([(2, 1)]), 0),  # Reversible single edge
        # 2. Identical raw PDAG inputs score 0
        (DAG([(1, 2), (2, 3)]), DAG([(1, 2), (2, 3)]), 0),
        (PDAG([("X", "Z"), ("Y", "Z")], []), PDAG([("X", "Z"), ("Y", "Z")], []), 0),
        # 3. Uncompleted/raw PDAGs are properly canonicalized via Meek's Rules before scoring
        (PDAG([("X", "Y")], [("Y", "Z")]), PDAG([("X", "Y"), ("Y", "Z")], []), 0),
        # 4. Standard structural errors (extra, missing, or improperly oriented edges) correctly accumulate penalties
        (
            DAG([("X", "Y"), ("Y", "Z")]),
            DAG([("X", "Z"), ("Y", "Z")]),
            3,
        ),  # Chain vs Collider
        (DAG([("X", "Z"), ("Y", "Z")]), missing_y_edge, 2),  # Compelled vs missing
        (
            DAG([("X", "Y"), ("Y", "Z")]),
            PDAG([("X", "Z"), ("Y", "Z")], []),
            3,
        ),  # DAG vs PDAG
        # 5. Edge cases: empty graphs and isolated nodes
        (DAG(), DAG(), 0),
        (isolated_3_a, isolated_3_b, 0),
    ],
)
def test_cpdagshd_scores(cpdag_scorer, graph1, graph2, expected_score):
    """Data-driven test covering equivalence, types, and scores."""
    # Ensure empty graphs have identical nodes
    if len(graph1.nodes()) == 0 and len(graph2.nodes()) == 0:
        graph1.add_nodes_from(["X", "Y", "Z"])
        graph2.add_nodes_from(["X", "Y", "Z"])

    assert cpdag_scorer(graph1, graph2) == expected_score


def test_cpdagshd_compelled_edge_vs_missing(cpdag_scorer):
    """Collider true graph vs estimated graph missing the Y→Z edge → 2."""
    true = DAG([("X", "Z"), ("Y", "Z")])
    est = DAG([("X", "Z")])
    est.add_node("Y")
    assert cpdag_scorer(true, est) == 2


def test_cpdagshd_isolated_nodes(cpdag_scorer):
    """Isolated nodes with same edge structure → 0, no crash."""
    dag1 = DAG([(1, 2)])
    dag1.add_node(3)
    dag2 = DAG([(1, 2)])
    dag2.add_node(3)
    assert cpdag_scorer(dag1, dag2) == 0


def test_cpdagshd_is_symmetric(cpdag_scorer):
    """CPDAGSHD(A, B) == CPDAGSHD(B, A)."""
    chain = DAG([("X", "Y"), ("Y", "Z")])
    collider = DAG([("X", "Z"), ("Y", "Z")])
    assert cpdag_scorer(chain, collider) == cpdag_scorer(collider, chain)


def test_cpdagshd_unequal_nodes_raises(cpdag_scorer):
    """Different node sets → ValueError with correct message."""
    dag1 = DAG([(1, 2)])
    dag2 = DAG([(3, 4)])
    with pytest.raises(ValueError, match=r"The graphs must have the same nodes\."):
        cpdag_scorer(dag1, dag2)
