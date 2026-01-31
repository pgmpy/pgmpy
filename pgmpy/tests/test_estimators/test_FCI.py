import numpy as np
import pandas as pd

from pgmpy.estimators.FCI import FCI
from pgmpy.independencies import Independencies


def fake_ci_oracle(X, Y, Z=None, **kwargs):
    Z = tuple(Z or [])

    independencies = {
        ("B", "C", ()),
        ("B", "D", ()),
        ("C", "D", ()),
        ("A", "B", ("C",)),
        ("A", "C", ("B",)),
    }

    return (X, Y, Z) in independencies or (Y, X, Z) in independencies


def collider_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=n)
    B = rng.normal(size=n)
    C = A + B + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


def fork_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=n)
    A = C + rng.normal(scale=0.1, size=n)
    B = C + rng.normal(scale=0.1, size=n)

    return pd.DataFrame({"A": A, "B": B, "C": C})


def chain_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=n)
    B = A + rng.normal(scale=0.1, size=n)
    C = B + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


def skeleton(pag):
    return {frozenset((u, v)) for u, v, *_ in pag.edges()}


def edge_points_to(pag, source, dest):
    return pag.get_edge_marks(source, dest)[1] == ">"


def test_collider():
    data = collider_data()
    pag = FCI(data).estimate(ci_test=fake_ci_oracle)

    assert skeleton(pag) == {frozenset(("A", "C")), frozenset(("B", "C"))}

    assert not edge_points_to(pag, "A", "C")
    assert not edge_points_to(pag, "B", "C")


def test_chain():
    data = chain_data()
    pag = FCI(data).estimate(ci_test=fake_ci_oracle)

    assert skeleton(pag) == {
        frozenset(("A", "B")),
        frozenset(("B", "C")),
    }

    assert not (edge_points_to(pag, "A", "B") and edge_points_to(pag, "C", "B"))


def test_from_independencies():
    ind = Independencies(["A", "B"], ["A", ["B"], "C"]).closure()

    pag = FCI(independencies=ind).estimate(ci_test="independence_match")

    assert skeleton(pag) == {
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    }
