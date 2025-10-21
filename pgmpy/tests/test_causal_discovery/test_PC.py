import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import PC
from pgmpy.independencies import Independencies
from pgmpy.models import DiscreteBayesianNetwork


def make_estimator():
    return PC()


@parametrize_with_checks([make_estimator()])
def test_pc_compatibility(estimator, check):
    check(estimator)


def fake_ci_t(X, Y, Z=[], **kwargs):
    """
    A mock CI testing function which gives False for every condition
    except for the following:
        1. B \u27c2 C
        2. B \u27c2 D
        3. C \u27c2 D
        4. A \u27c2 B | C
        5. A \u27c2 C | B
    """
    Z = list(Z)
    if X == "B":
        if Y == "C" or Y == "D":
            return True
        elif Y == "A" and Z == ["C"]:
            return True
    elif X == "C" and Y == "D" and Z == []:
        return True
    elif X == "D" and Y == "C" and Z == []:
        return True
    elif Y == "B":
        if X == "C" or X == "D":
            return True
        elif X == "A" and Z == ["C"]:
            return True
    elif X == "A" and Y == "C" and Z == ["B"]:
        return True
    elif X == "C" and Y == "A" and Z == ["B"]:
        return True
    return False


@pytest.fixture
def fake_data():
    np.random.seed(42)
    return pd.DataFrame(np.random.random((1000, 4)), columns=["A", "B", "C", "D"])


@pytest.mark.parametrize("variant", ["orig", "stable"])
def test_build_skeleton(fake_data, variant):
    skel, sep_set = PC()._build_skeleton(fake_data, ci_test=fake_ci_t, variant=variant)
    expected_edges = {("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)

    # Test with 0 conditional vars
    skel, sep_set = PC()._build_skeleton(
        fake_data,
        ci_test=fake_ci_t,
        max_cond_vars=0,
        variant=variant,
    )
    expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_from_ind(variant):
    # Specify a set of independencies
    ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
    ind = ind.closure()

    cols_ = ["A", "B", "C", "D"]
    rows = [[0, 0, 0, 0], [0, 0, 0, 0]]
    data = pd.DataFrame(data=rows, columns=cols_)
    estimator = PC(
        variant=variant,
        ci_test="independence_match",
        return_type="skeleton",
        n_jobs=2,
        show_progress=False,
    )
    estimator.fit(
        data,
        independencies=ind,
    )

    expected_edges = {("A", "D"), ("B", "D"), ("C", "D")}
    expected_sepsets = {
        frozenset(("A", "C")): tuple(),
        frozenset(("A", "B")): tuple(),
        frozenset(("C", "B")): tuple(),
    }
    for u, v in estimator.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert estimator.separating_sets_ == expected_sepsets

    # Generate independencies from a model.
    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
    estimator = PC(
        variant=variant,
        ci_test="independence_match",
        return_type="skeleton",
        n_jobs=2,
        show_progress=False,
    )

    cols_ = ["A", "B", "C", "D", "E"]
    rows = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    data = pd.DataFrame(data=rows, columns=cols_)
    estimator.fit(
        X=data,
        independencies=model.get_independencies(),
    )

    expected_edges = estimator.graph_.edges()
    expected_sepsets1 = {
        frozenset(("D", "C")): ("B",),
        frozenset(("E", "B")): ("C",),
        frozenset(("A", "D")): tuple(),
        frozenset(("E", "D")): ("C",),
        frozenset(("E", "A")): ("C",),
        frozenset(("A", "B")): tuple(),
    }
    expected_sepsets2 = {
        frozenset(("D", "C")): ("B",),
        frozenset(("E", "B")): ("C",),
        frozenset(("A", "D")): tuple(),
        frozenset(("E", "D")): ("B",),
        frozenset(("E", "A")): ("C",),
        frozenset(("A", "B")): tuple(),
    }
    for u, v in estimator.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)

    assert (estimator.separating_sets_ == expected_sepsets1) or (
        estimator.separating_sets_ == expected_sepsets2
    )


def test_skeleton_to_pdag():
    # D - A - C - B  ==> D - A -> C <- B
    skel = nx.Graph([("A", "D"), ("A", "C"), ("B", "C")])
    sep_sets = {
        frozenset({"D", "C"}): ("A",),
        frozenset({"A", "B"}): tuple(),
        frozenset({"D", "B"}): ("A",),
    }
    pdag = PC()._orient_colliders(skel, sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == set([("B", "C"), ("A", "D"), ("A", "C"), ("D", "A")])

    # C - A - B  ==> C -> A <- B
    skel = nx.Graph([("A", "B"), ("A", "C")])
    sep_sets = {frozenset({"B", "C"}): ()}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == set([("B", "A"), ("C", "A")])

    # C - A - B ==> C - A - B
    skel = nx.Graph([("A", "B"), ("A", "C")])
    sep_sets = {frozenset({"B", "C"}): ("A",)}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == set([("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")])

    # {A, B} - C - D ==> {A, B} -> C -> D
    skel = nx.Graph([("A", "C"), ("B", "C"), ("C", "D")])
    sep_sets = {
        frozenset({"A", "B"}): tuple(),
        frozenset({"A", "D"}): ("C",),
        frozenset({"B", "D"}): ("C",),
    }
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == set([("A", "C"), ("B", "C"), ("C", "D")])

    # C - A - B - {C, D} ==> C <- A -> B <- D; B -> C
    skel = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")])
    sep_sets = {frozenset({"A", "D"}): tuple(), frozenset({"C", "D"}): ("A", "B")}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == set([("A", "B"), ("B", "C"), ("A", "C"), ("D", "B")])

    skel = nx.Graph([("A", "B"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")])
    sep_sets = {frozenset({"A", "C"}): ("B",)}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == set(
        [
            ("A", "B"),
            ("B", "A"),
            ("B", "C"),
            ("C", "B"),
            ("A", "D"),
            ("B", "D"),
            ("C", "D"),
        ]
    )


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_estimate_dag(variant):
    ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
    ind = ind.closure()

    cols_ = ["A", "B", "C", "D"]
    rows = [[0, 0, 0, 0], [0, 0, 0, 0]]
    data = pd.DataFrame(data=rows, columns=cols_)

    estimator = PC(
        variant="orig",
        ci_test="independence_match",
        return_type="dag",
        n_jobs=2,
        show_progress=False,
    ).fit(data, independencies=ind)

    expected_edges = {("B", "D"), ("A", "D"), ("C", "D")}
    assert estimator.graph_.edges() == expected_edges

    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
    cols_ = ["A", "B", "C", "D", "E"]
    rows = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    data = pd.DataFrame(data=rows, columns=cols_)

    estimator = PC(
        variant="orig",
        ci_test="independence_match",
        return_type="dag",
        n_jobs=2,
        show_progress=False,
    ).fit(data, independencies=model.get_independencies())

    expected_edges_1 = set(estimator.graph_.edges())
    expected_edges_2 = {("B", "C"), ("A", "C"), ("C", "E"), ("D", "B")}
    assert (set(estimator.graph_.edges()) == expected_edges_1) or (
        set(estimator.graph_.edges()) == expected_edges_2
    )


@pytest.fixture(autouse=True)
def cleanup_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
