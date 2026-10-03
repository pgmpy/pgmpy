import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.base import ADMG, DAG, PDAG
from pgmpy.causal_discovery import PC
from pgmpy.metrics import SHD, SelfCompatibilityScore
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_example_model


def all_seps_agree(dag_one: nx.DiGraph, dag_two: nx.DiGraph):
    dir, undir = [], []
    for u, v in dag_one.edges():
        if (v, u) in dag_one.edges():
            if u>v:
                undir.append((u, v))
        else:
            dir.append((u, v))
    admg_one = ADMG(directed_ebunch=dir, bidirected_ebunch=undir)

    dir, undir = [], []
    for u, v in dag_two.edges():
        if (v, u) in dag_two.edges():
            if u>v:
                undir.append((u, v))
        else:
            dir.append((u, v))

    admg_two = ADMG(directed_ebunch=dir, bidirected_ebunch=undir)

    seps_agree = []
    for x, y in [(x, y) for x in admg_one.nodes for y in admg_one.nodes if x != y]:
        for z in [set()] + [{z} for z in admg_one.nodes if z != x and z != y]:
            original_sep = admg_one.is_mseparated({x}, {y}, set(z))
            sep_after = admg_two.is_mseparated({x}, {y}, set(z))
            seps_agree.append(original_sep == sep_after)
    return all(seps_agree)


def test_marginalize_returns_expected_graph_for_m_graph():
    dag = DAG()
    dag.add_edge("a", "x")
    dag.add_edge("a", "b")
    dag.add_edge("c", "b")
    dag.add_edge("c", "y")

    subset = {"x", "y"}
    marginal_admg = SelfCompatibilityScore()._latent_admg(dag, subset)
    assert set(marginal_admg.nodes()) == subset
    assert all_seps_agree(marginal_admg, dag)

    expected_dag = PDAG()
    expected_dag.add_nodes_from(["x", "y"])
    assert SHD().evaluate(marginal_admg, expected_dag) == 0

    expected_dag.add_edge("x", "y")
    assert SHD().evaluate(marginal_admg, expected_dag) != 0


def test_marginalize_returns_expected_graph_for_chain_graph():
    dag = DAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    dag.add_edge("c", "d")

    subset = {"a", "c"}
    marginal_admg = SelfCompatibilityScore()._latent_admg(dag, subset)

    assert set(marginal_admg.nodes()) == subset
    assert all_seps_agree(marginal_admg, dag)

    expected_dag = PDAG()
    expected_dag.add_edge("a", "c")
    assert SHD().evaluate(marginal_admg, expected_dag) == 0

    expected_dag.remove_edge("a", "c")
    assert SHD().evaluate(marginal_admg, expected_dag) != 0


def test_marginalize_returns_expected_graph_for_unshielded_collider_graph():
    dag = DAG()
    dag.add_edge("a", "b")
    dag.add_edge("c", "b")
    dag.add_edge("d", "c")

    subset = {"a", "c"}
    marginal_admg = SelfCompatibilityScore()._latent_admg(dag, subset)

    assert set(marginal_admg.nodes()) == subset
    assert all_seps_agree(marginal_admg, dag)

    expected_dag = PDAG()
    expected_dag.add_nodes_from(["a", "c"])
    assert SHD().evaluate(marginal_admg, expected_dag) == 0

    expected_dag.add_edge("a", "c")
    assert SHD().evaluate(marginal_admg, expected_dag) != 0


def test_marginalize_returns_expected_graph_for_pure_confounding_graph():
    dag = DAG()
    dag.add_edge("a", "b")
    dag.add_edge("a", "c")

    subset = {"b", "c"}
    marginal_admg = SelfCompatibilityScore()._latent_admg(dag, subset)

    assert set(marginal_admg.nodes()) == subset
    print(marginal_admg.edges)
    assert all_seps_agree(marginal_admg, dag)

    expected_dag = PDAG()
    expected_dag.add_nodes_from(["b", "c"])
    expected_dag.add_edge("b", "c")
    expected_dag.add_edge("c", "b")
    assert SHD().evaluate(marginal_admg, expected_dag) == 0

    expected_dag.remove_edge("b", "c")
    assert SHD().evaluate(marginal_admg, expected_dag) != 0


def test_marginalize_returns_expected_graph_for_indirect_confounding():
    dag = DAG()
    dag.add_edge("u", "a")
    dag.add_edge("u", "b")

    subset = {"a", "b"}
    marginal_admg = SelfCompatibilityScore()._latent_admg(dag, subset)

    assert set(marginal_admg.nodes()) == subset
    assert all_seps_agree(marginal_admg, dag)

    expected_dag = PDAG()
    expected_dag.add_nodes_from(["a", "b"])
    expected_dag.add_edge("a", "b")
    expected_dag.add_edge("b", "a")

    print(marginal_admg.edges)
    assert SHD().evaluate(marginal_admg, expected_dag) == 0

    expected_dag.remove_edge("a", "b")
    assert SHD().evaluate(marginal_admg, expected_dag) != 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_dag():
    return DiscreteBayesianNetwork([("A", "B"), ("B", "C")])


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "A": rng.randint(2, size=100),
            "B": rng.randint(2, size=100),
            "C": rng.randint(2, size=100),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_perfect_compatibility(full_dag, data):
    """
    If joint and all marginals always return the true DAG:
    SHD for each subset = 0 ⇒ mean = 0.0.
    """

    class PerfectEstimator:
        def fit(self, X):
            self.causal_graph_ = full_dag
            return self

    score = SelfCompatibilityScore(num_subsets=10, subset_fraction=0.5, seed=1).evaluate(data, PerfectEstimator)

    assert score == 0.0


def test_flip_compatibility(full_dag, data):
    """
    When the joint fit is correct but every marginal is flipped,
    each subset's SHD should be 2 (two edge reversals), so the mean = 2.0.
    """
    full_edges = list(full_dag.edges())
    state = {"calls": 0}

    class FlipEstimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            state["calls"] += 1
            cols = list(X.columns)
            dag = DiscreteBayesianNetwork([])
            for node in cols:
                dag.add_node(node)
            if state["calls"] == 1:
                for u, v in full_edges:
                    dag.add_edge(u, v)
            else:
                for u, v in full_edges:
                    dag.add_edge(v, u)
            self.causal_graph_ = dag
            return self

    score = SelfCompatibilityScore(num_subsets=10, subset_fraction=1.0, seed=0).evaluate(data, FlipEstimator)
    assert score == pytest.approx(2.0)


def test_kwargs_forwarded(data):
    """
    Ensure that arbitrary kwargs (e.g. alpha, foo) are passed through
    exactly once to the estimator when no marginal runs occur (num_subsets=0).
    """
    seen = {}

    class RecordEstimator:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fit(self, X):
            self.causal_graph_ = DiscreteBayesianNetwork([])
            return self

    SelfCompatibilityScore(num_subsets=0, subset_fraction=1.0, seed=3, alpha=0.01, foo="bar").evaluate(
        data, RecordEstimator
    )
    assert "alpha" in seen
    assert seen["alpha"] == 0.01
    assert "foo" in seen
    assert seen["foo"] == "bar"


def test_perfect_subset_projection():
    """
    If both joint and marginal estimators always return the exact
    Definition-5 projection of the true model onto S, then SHD=0.
    """
    true_model = DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("C", "D")])

    class ProjEstimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            S = set(X.columns)
            g = SelfCompatibilityScore()._latent_admg(true_model, list(S))
            m = DiscreteBayesianNetwork([])
            m.add_nodes_from(g.nodes())
            for u, v in g.edges():
                m.add_edge(u, v)
            self.causal_graph_ = m
            return self

    dummy = pd.DataFrame(
        {
            "A": np.zeros(50),
            "B": np.zeros(50),
            "C": np.zeros(50),
            "D": np.zeros(50),
        }
    )

    score = SelfCompatibilityScore(num_subsets=20, subset_fraction=0.75, seed=42).evaluate(dummy, ProjEstimator)
    assert score == 0.0


def test_child_example_low_score():
    """
    When fitting the small Child network on its own simulated data,
    the self-compatibility score should be very low.
    """
    model = get_example_model("child")
    data = model.simulate(n_samples=500, seed=42)

    score = SelfCompatibilityScore(
        num_subsets=40,
        subset_fraction=0.8,
        seed=42,
        return_type='dag',
    ).evaluate(data, PC)

    assert score > 0.0, "SHD should be positive"
    assert score < 10.0, "SHD should be less than 10"
