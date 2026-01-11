import numpy as np
import pandas as pd

from pgmpy.estimators.FCI import FCI
from pgmpy.independencies.Independencies import Independencies


def skeleton(pag):
    """Return undirected skeleton as frozensets of node pairs."""
    return {frozenset((u, v)) for u, v, _, _ in pag.edges()}


def mark(pag, u, v):
    """
    Direction-agnostic access to PAG edge marks.
    Always returns (mark_u, mark_v) in the order requested.
    """
    return pag.get_edge_marks(u, v)


def test_estimate_collider_structure():
    np.random.seed(42)
    n = 5000

    A = np.random.randn(n)
    B = np.random.randn(n)
    C = A + B + np.random.randn(n) * 0.1

    data = pd.DataFrame({"A": A, "B": B, "C": C})
    pag = FCI(data).estimate(ci_test="pearsonr")

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }

    assert mark(pag, "A", "C")[1] == ">"
    assert mark(pag, "B", "C")[1] == ">"


def test_estimate_fork_structure():
    np.random.seed(42)
    n = 5000

    C = np.random.randn(n)
    A = C + np.random.randn(n) * 0.1
    B = C + np.random.randn(n) * 0.1

    data = pd.DataFrame({"A": A, "B": B, "C": C})
    pag = FCI(data).estimate(ci_test="pearsonr")

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }

    assert mark(pag, "A", "C")[1] != ">"
    assert mark(pag, "B", "C")[1] != ">"


def test_estimate_chain_structure():
    np.random.seed(42)
    n = 5000

    A = np.random.randn(n)
    B = A + np.random.randn(n) * 0.1
    C = B + np.random.randn(n) * 0.1

    data = pd.DataFrame({"A": A, "B": B, "C": C})
    pag = FCI(data).estimate(ci_test="pearsonr")

    assert skeleton(pag) == {
        frozenset(("A", "B")),
        frozenset(("B", "C")),
    }

    assert not (mark(pag, "A", "B")[1] == ">" and mark(pag, "C", "B")[1] == ">")


def test_estimate_latent_confounder():
    np.random.seed(42)
    n = 6000

    L = np.random.randn(n)
    X = L + np.random.randn(n) * 0.1
    Y = L + np.random.randn(n) * 0.1

    data = pd.DataFrame({"X": X, "Y": Y})
    pag = FCI(data).estimate(ci_test="pearsonr")

    assert skeleton(pag) == {
        frozenset(("X", "Y")),
    }

    assert mark(pag, "X", "Y") == ("o", "o")


def test_estimate_from_independencies():
    ind = Independencies(
        ["A", "B"],
        ["A", ["B"], "C"],
    ).closure()

    pag = FCI(independencies=ind).estimate(ci_test="independence_match")

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }
