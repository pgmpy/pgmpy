import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import CPDAGSHD


@pytest.fixture
def cpdag_scorer():
    return CPDAGSHD()


# -----------------------------------------------------------------------
# GROUP 1: Core MEC-awareness — the fundamental property of this metric.
# Markov-equivalent DAGs must score 0; SHD would give non-zero.
# -----------------------------------------------------------------------


def test_cpdagshd_equivalent_dags_score_zero(cpdag_scorer):
    """Chain and fork are Markov equivalent → CPDAGSHD = 0, SHD would give 1."""
    chain = DAG([("X", "Y"), ("Y", "Z")])
    fork = DAG([("Y", "X"), ("Y", "Z")])
    assert cpdag_scorer(chain, fork) == 0


def test_cpdagshd_identical_dags(cpdag_scorer):
    """Identical DAGs → CPDAGSHD = 0."""
    dag = DAG([(1, 2), (2, 3)])
    assert cpdag_scorer(dag, dag) == 0


def test_cpdagshd_non_equivalent_dags_positive(cpdag_scorer):
    """Chain vs collider are non-equivalent → CPDAGSHD > 0."""
    chain = DAG([("X", "Y"), ("Y", "Z")])
    collider = DAG([("X", "Z"), ("Y", "Z")])
    assert cpdag_scorer(chain, collider) > 0


# -----------------------------------------------------------------------
# GROUP 2: Known exact values — traced and executed, do not change.
# -----------------------------------------------------------------------


def test_cpdagshd_chain_vs_collider_exact(cpdag_scorer):
    """chain X→Y→Z vs collider X→Z←Y: all 3 node pairs differ → 3.

    X-Y: undirected in chain-CPDAG, none in collider-CPDAG → mismatch
    X-Z: none in chain-CPDAG,       X→Z in collider-CPDAG  → mismatch
    Y-Z: undirected in chain-CPDAG, Y→Z in collider-CPDAG  → mismatch
    """
    chain = DAG([("X", "Y"), ("Y", "Z")])
    collider = DAG([("X", "Z"), ("Y", "Z")])
    assert cpdag_scorer(chain, collider) == 3


def test_cpdagshd_reversed_reversible_edge(cpdag_scorer):
    """Reversing a reversible edge keeps both DAGs in the same MEC → 0.

    For a 2-node graph, the single edge is always reversible (no v-structures
    can form), so both DAG([(1,2)]) and DAG([(2,1)]) map to the same CPDAG
    with one undirected edge.
    """
    dag1 = DAG([(1, 2)])
    dag2 = DAG([(2, 1)])
    assert cpdag_scorer(dag1, dag2) == 0


def test_cpdagshd_compelled_edge_vs_missing(cpdag_scorer):
    """Collider true graph vs estimated graph missing the Y→Z edge → 2."""
    true = DAG([("X", "Z"), ("Y", "Z")])
    est = DAG([("X", "Z")])
    est.add_node("Y")
    assert cpdag_scorer(true, est) == 2


# -----------------------------------------------------------------------
# GROUP 3: PDAG input — metric must accept PDAG (default PC/GES output).
# -----------------------------------------------------------------------


def test_cpdagshd_pdag_est_input(cpdag_scorer):
    """A PDAG estimated graph is accepted without conversion crash."""
    pdag = PDAG(directed_ebunch=[("X", "Z"), ("Y", "Z")], undirected_ebunch=[])
    dag = DAG([("X", "Y"), ("Y", "Z")])
    result = cpdag_scorer(dag, pdag)
    assert result == 3


def test_cpdagshd_both_pdag_inputs(cpdag_scorer):
    """Two identical PDAG inputs → 0."""
    pdag1 = PDAG(directed_ebunch=[("X", "Z"), ("Y", "Z")], undirected_ebunch=[])
    pdag2 = PDAG(directed_ebunch=[("X", "Z"), ("Y", "Z")], undirected_ebunch=[])
    assert cpdag_scorer(pdag1, pdag2) == 0


def test_cpdagshd_uncompleted_pdag_input(cpdag_scorer):
    """An uncompleted PDAG where Meek's rules will force orientation.

    If X -> Y - Z and X,Z non-adjacent, Meek's Rule 1 forces Y -> Z.
    Therefore, PDAG(dir=[X->Y], undir=[Y-Z]) should match PDAG(dir=[X->Y, Y->Z]).
    """
    pdag_partial = PDAG(directed_ebunch=[("X", "Y")], undirected_ebunch=[("Y", "Z")])
    pdag_completed = PDAG(
        directed_ebunch=[("X", "Y"), ("Y", "Z")], undirected_ebunch=[]
    )
    assert cpdag_scorer(pdag_partial, pdag_completed) == 0


# -----------------------------------------------------------------------
# GROUP 4: Symmetry.
# -----------------------------------------------------------------------


def test_cpdagshd_is_symmetric(cpdag_scorer):
    """CPDAGSHD(A, B) == CPDAGSHD(B, A)."""
    chain = DAG([("X", "Y"), ("Y", "Z")])
    collider = DAG([("X", "Z"), ("Y", "Z")])
    assert cpdag_scorer(chain, collider) == cpdag_scorer(collider, chain)


# -----------------------------------------------------------------------
# GROUP 5: Edge cases.
# -----------------------------------------------------------------------


def test_cpdagshd_empty_graphs(cpdag_scorer):
    """Both graphs have no edges → CPDAGSHD = 0."""
    true = DAG()
    true.add_nodes_from(["X", "Y", "Z"])
    est = DAG()
    est.add_nodes_from(["X", "Y", "Z"])
    assert cpdag_scorer(true, est) == 0


def test_cpdagshd_isolated_nodes(cpdag_scorer):
    """Isolated nodes with same edge structure → 0, no crash."""
    dag1 = DAG([(1, 2)])
    dag1.add_node(3)
    dag2 = DAG([(1, 2)])
    dag2.add_node(3)
    assert cpdag_scorer(dag1, dag2) == 0


# -----------------------------------------------------------------------
# GROUP 6: Input validation.
# -----------------------------------------------------------------------


def test_cpdagshd_unequal_nodes_raises(cpdag_scorer):
    """Different node sets → ValueError with correct message."""
    dag1 = DAG([(1, 2)])
    dag2 = DAG([(3, 4)])
    with pytest.raises(ValueError, match=r"The graphs must have the same nodes\."):
        cpdag_scorer(dag1, dag2)
