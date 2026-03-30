import logging

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery import PC
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator
from pgmpy.example_models import load_model
from pgmpy.independencies import Independencies
from pgmpy.metrics import SHD, CorrelationScore
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [PC(return_type="dag", show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
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
    skel, _ = PC()._build_skeleton(fake_data, ci_test=fake_ci_t, variant=variant)
    expected_edges = {("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)

    # Test with 0 conditional vars
    skel, _ = PC()._build_skeleton(
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
        return_type="pdag",
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
        return_type="pdag",
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

    expected_edges = estimator.causal_graph_.edges()
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

    assert (estimator.separating_sets_ == expected_sepsets1) or (estimator.separating_sets_ == expected_sepsets2)


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
    assert set(pdag.edges()) == {("B", "C"), ("A", "D"), ("A", "C"), ("D", "A")}

    # C - A - B  ==> C -> A <- B
    skel = nx.Graph([("A", "B"), ("A", "C")])
    sep_sets = {frozenset({"B", "C"}): ()}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == {("B", "A"), ("C", "A")}

    # C - A - B ==> C - A - B
    skel = nx.Graph([("A", "B"), ("A", "C")])
    sep_sets = {frozenset({"B", "C"}): ("A",)}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")}

    # {A, B} - C - D ==> {A, B} -> C -> D
    skel = nx.Graph([("A", "C"), ("B", "C"), ("C", "D")])
    sep_sets = {
        frozenset({"A", "B"}): tuple(),
        frozenset({"A", "D"}): ("C",),
        frozenset({"B", "D"}): ("C",),
    }
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == {("A", "C"), ("B", "C"), ("C", "D")}

    # C - A - B - {C, D} ==> C <- A -> B <- D; B -> C
    skel = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")])
    sep_sets = {frozenset({"A", "D"}): tuple(), frozenset({"C", "D"}): ("A", "B")}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == {("A", "B"), ("B", "C"), ("A", "C"), ("D", "B")}

    skel = nx.Graph([("A", "B"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")])
    sep_sets = {frozenset({"A", "C"}): ("B",)}
    pdag = PC()._orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == {
        ("A", "B"),
        ("B", "A"),
        ("B", "C"),
        ("C", "B"),
        ("A", "D"),
        ("B", "D"),
        ("C", "D"),
    }


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
    assert estimator.causal_graph_.edges() == expected_edges

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

    expected_edges_1 = set(estimator.causal_graph_.edges())
    expected_edges_2 = {("B", "C"), ("A", "C"), ("C", "E"), ("D", "B")}
    assert (set(estimator.causal_graph_.edges()) == expected_edges_1) or (
        set(estimator.causal_graph_.edges()) == expected_edges_2
    )


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_chi_square(variant):

    # Fake dataset no: 1
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    est = PC(
        variant=variant,
        ci_test="chi_square",
        return_type="pdag",
        significance_level=0.005,
        show_progress=False,
    )
    est.fit(X=data)
    expected_edges = {("A", "F"), ("B", "F"), ("C", "F")}
    expected_sepsets = {
        frozenset(("D", "F")): tuple(),
        frozenset(("D", "B")): tuple(),
        frozenset(("A", "C")): tuple(),
        frozenset(("D", "E")): tuple(),
        frozenset(("E", "F")): tuple(),
        frozenset(("E", "C")): tuple(),
        frozenset(("E", "B")): tuple(),
        frozenset(("D", "C")): tuple(),
        frozenset(("A", "B")): tuple(),
        frozenset(("A", "E")): tuple(),
        frozenset(("B", "C")): tuple(),
        frozenset(("A", "D")): tuple(),
    }
    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert est.separating_sets_ == expected_sepsets

    # Fake dataset no: 2 Expected structure X <- Z -> Y
    def fake_ci(X, Y, Z=tuple(), **kwargs):
        if X == "X" and Y == "Y" and Z == ("Z",):
            return True
        elif X == "Y" and Y == "X" and Z == ("Z",):
            return True
        else:
            return False

    np.random.seed(42)
    fake_data = pd.DataFrame(
        np.random.randint(low=0, high=2, size=(10000, 3)),
        columns=["X", "Y", "Z"],
    )
    est = PC(
        variant=variant,
        ci_test=fake_ci,
        return_type="pdag",
        show_progress=False,
    )
    est.fit(X=fake_data)
    expected_edges = {("X", "Z"), ("Y", "Z")}
    expected_sepsets = {frozenset(("X", "Y")): ("Z",)}
    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert est.separating_sets_ == expected_sepsets


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_discrete(variant):
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]

    for test in [
        "g_sq",
        "log_likelihood",
        "modified_log_likelihood",
        "power_divergence",
    ]:
        est = PC(
            variant=variant,
            ci_test=test,
            return_type="pdag",
            significance_level=0.005,
            n_jobs=2,
            show_progress=False,
        )
        est.fit(X=data)


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_dag_discrete(variant):

    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 3, size=(10000, 3)), columns=list("XYZ"))
    data["sum"] = data.sum(axis=1)
    est = PC(
        variant=variant,
        ci_test="chi_square",
        return_type="dag",
        significance_level=0.001,
        n_jobs=2,
        show_progress=False,
    )
    est.fit(X=data)
    expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
    assert set(est.causal_graph_.edges()) == expected_edges


def test_search_space():
    adult_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    search_space = [
        ("Age", "Education"),
        ("Education", "HoursPerWeek"),
        ("Education", "Income"),
        ("HoursPerWeek", "Income"),
        ("Age", "Income"),
    ]

    expert_knowledge = ExpertKnowledge(search_space=search_space)

    est = PC(
        expert_knowledge=expert_knowledge,
        enforce_expert_knowledge=True,
        show_progress=False,
    )

    est.fit(X=adult_data)
    # assert if dag is a subset of search_space
    for edge in est.causal_graph_.edges():
        assert edge in search_space


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
@pytest.mark.parametrize("ci_test", ["pearsonr", "pillai", "gcm"])
@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_continuous(ci_test, variant):

    # Fake dataset no: 1
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 5), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    est = PC(
        variant=variant,
        ci_test=ci_test,
        return_type="pdag",
        n_jobs=2,
        show_progress=False,
    )
    est.fit(X=data)
    expected_edges = {("A", "F"), ("B", "F"), ("C", "F")}
    expected_edges_stable = {("A", "F"), ("B", "C"), ("B", "F"), ("C", "F")}
    expected_sepsets = {
        frozenset(("D", "F")): tuple(),
        frozenset(("D", "B")): tuple(),
        frozenset(("A", "C")): tuple(),
        frozenset(("D", "E")): tuple(),
        frozenset(("E", "F")): tuple(),
        frozenset(("E", "C")): tuple(),
        frozenset(("E", "B")): tuple(),
        frozenset(("D", "C")): tuple(),
        frozenset(("A", "B")): tuple(),
        frozenset(("A", "E")): tuple(),
        frozenset(("B", "C")): tuple(),
        frozenset(("A", "D")): tuple(),
        # This one is only for stable version.
        frozenset(("C", "B")): tuple(),
    }
    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges_stable) or ((v, u) in expected_edges_stable)

    for key, value in est.separating_sets_.items():
        assert est.separating_sets_[key] == expected_sepsets[key]

    # Fake dataset no: 2. Expected model structure X <- Z -> Y
    def fake_ci(X, Y, Z=tuple(), **kwargs):
        if X == "X" and Y == "Y" and Z == ("Z",):
            return True
        elif X == "Y" and Y == "X" and Z == ("Z",):
            return True
        else:
            return False

    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
    est = PC(
        variant=variant,
        ci_test=fake_ci,
        return_type="pdag",
        n_jobs=2,
        show_progress=False,
    )
    est.fit(X=data)
    expected_edges = {("X", "Z"), ("Y", "Z")}
    expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

    for u, v in est.skeleton_.edges():
        assert ((u, v) in expected_edges) or ((v, u) in expected_edges)
    assert est.separating_sets_ == expected_sepsets


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
@pytest.mark.parametrize("ci_test", ["pearsonr", "pillai", "gcm"])
@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_dag_continuous(ci_test, variant):
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
    data["sum"] = data.sum(axis=1)
    est = PC(
        variant=variant,
        ci_test=ci_test,
        return_type="dag",
        n_jobs=2,
        show_progress=False,
    )
    est.fit(X=data)

    expected_edges = {("Z", "sum"), ("X", "sum"), ("Y", "sum")}
    assert set(est.causal_graph_.edges()) == expected_edges


def test_pc_alarm():
    alarm_model = load_model("bnlearn/alarm")
    data = BayesianModelSampling(alarm_model).forward_sample(size=int(1e4), seed=42)
    est = PC(variant="stable", max_cond_vars=5, n_jobs=2, show_progress=False)
    est.fit(X=data)


def test_pc_asia(caplog):
    asia_model = load_model("bnlearn/asia")
    data = asia_model.simulate(n_samples=int(1e5), seed=42)
    req_edges = [("xray", "either")]
    background = ExpertKnowledge(required_edges=req_edges)
    est = PC(
        variant="stable",
        max_cond_vars=4,
        expert_knowledge=background,
        n_jobs=2,
        show_progress=False,
    )

    pgmpy_logger = logging.getLogger("pgmpy")
    pgmpy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING", logger="pgmpy"):
            est.fit(X=data)
    finally:
        pgmpy_logger.removeHandler(caplog.handler)
    expected_warning = (
        "Specified expert knowledge conflicts with learned structure. Ignoring edge xray->either from required edges"
    )

    assert any(expected_warning in message for message in caplog.messages)


def test_pc_asia_expert():
    asia_model = load_model("bnlearn/asia")
    data = asia_model.simulate(n_samples=int(1e5), seed=42)
    est = PC(
        variant="stable",
        max_cond_vars=2,
        expert_knowledge=ExpertKnowledge(
            required_edges=[
                ("lung", "either"),
                ("tub", "either"),
                ("bronc", "dysp"),
            ]
        ),
        n_jobs=2,
        show_progress=False,
    )
    est.fit(X=data)
    pdag = est.causal_graph_
    if ("lung", "either") in pdag.edges() or ("either", "lung") in pdag.edges():
        assert ("lung", "either") in pdag.directed_edges
    if ("tub", "either") in pdag.edges() or ("either", "tub") in pdag.edges():
        assert ("tub", "either") in pdag.directed_edges
    if ("bronc", "dysp") in pdag.edges() or ("dysp", "bronc") in pdag.edges():
        assert ("bronc", "dysp") in pdag.directed_edges


def test_temporal_pc_cancer():
    cancer_model = load_model("bnlearn/cancer")
    data = cancer_model.simulate(n_samples=int(5e4), seed=42)

    background = ExpertKnowledge(  # e.g. we know only "Pollution", "Smoker", "Cancer" can be the causes of others
        temporal_order=[["Pollution", "Smoker", "Cancer"], ["Dyspnoea", "Xray"]],
        max_cond_vars=4,
    )
    est = PC(
        variant="stable",
        expert_knowledge=background,
        n_jobs=2,
        show_progress=False,
    )
    est.fit(X=data)
    pdag = est.causal_graph_
    assert set(pdag.edges()) == {
        ("Cancer", "Xray"),
        ("Cancer", "Dyspnoea"),
        ("Smoker", "Cancer"),
        ("Pollution", "Cancer"),
    }


def test_temporal_pc_sachs():
    temporal_order = [
        ["PKC", "Plcg"],
        [
            "PKA",
            "Raf",
            "Jnk",
            "P38",
            "PIP3",
            "PIP2",
            "Mek",
            "Erk",
        ],
        ["Akt"],
    ]
    temporal_forbidden_edges = {
        ("PKA", "PKC"),
        ("PKA", "Plcg"),
        ("Raf", "PKC"),
        ("Raf", "Plcg"),
        ("Jnk", "PKC"),
        ("Jnk", "Plcg"),
        ("P38", "PKC"),
        ("P38", "Plcg"),
        ("PIP3", "PKC"),
        ("PIP3", "Plcg"),
        ("PIP2", "PKC"),
        ("PIP2", "Plcg"),
        ("Mek", "PKC"),
        ("Mek", "Plcg"),
        ("Erk", "PKC"),
        ("Erk", "Plcg"),
        ("Akt", "PKC"),
        ("Akt", "Plcg"),
        ("Akt", "PKA"),
        ("Akt", "Raf"),
        ("Akt", "Jnk"),
        ("Akt", "P38"),
        ("Akt", "PIP3"),
        ("Akt", "PIP2"),
        ("Akt", "Mek"),
        ("Akt", "Erk"),
    }

    model = load_model("bnlearn/sachs")
    df = model.simulate(int(1e3))

    expert = ExpertKnowledge(temporal_order=temporal_order)
    pdag = PC(ci_test="chi_square", expert_knowledge=expert).fit(X=df).causal_graph_
    assert temporal_forbidden_edges.isdisjoint(set(pdag.edges()))


def _fake_ci_temporal(X, Y, Z=[], **kwargs):
    Z = list(Z)
    pair = frozenset((X, Y))
    if pair == frozenset(("A", "B")) and sorted(Z) == ["C"]:
        return True
    return False


@pytest.mark.parametrize("estimator_class", [PC, BaseConstraintEstimator])
def test_temporal_ordering_sepsets_and_skeleton(estimator_class):
    graph = UndirectedGraph([("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")])
    temporal_ordering = {"A": 3, "B": 1, "C": 2, "D": 0}

    result_ab = sorted(
        estimator_class._get_potential_sepsets(
            u="A",
            v="B",
            temporal_ordering=temporal_ordering,
            graph=graph,
            lim_neighbors=1,
        )
    )
    assert result_ab == [("D",), ("D",)]

    # Symmetry: swapping u and v gives the same separating sets.
    result_ba = sorted(
        estimator_class._get_potential_sepsets(
            u="B",
            v="A",
            temporal_ordering=temporal_ordering,
            graph=graph,
            lim_neighbors=1,
        )
    )
    assert result_ab == result_ba

    # Without temporal ordering, all neighbors are candidates: C and D from both sides.
    result_no_temporal = sorted(
        estimator_class._get_potential_sepsets(
            u="A",
            v="B",
            temporal_ordering={},
            graph=graph,
            lim_neighbors=1,
        )
    )
    assert result_no_temporal == [("C",), ("C",), ("D",), ("D",)]

    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(100, 4)), columns=["A", "B", "C", "D"])
    expert = ExpertKnowledge(temporal_order=[["D"], ["B"], ["C"], ["A"]])

    if estimator_class is PC:
        skel, _ = PC(
            variant="stable",
            ci_test=_fake_ci_temporal,
            expert_knowledge=expert,
            show_progress=False,
        )._build_skeleton(
            data,
            ci_test=_fake_ci_temporal,
            expert_knowledge=expert,
            variant="stable",
            show_progress=False,
        )
    else:
        skel, _ = BaseConstraintEstimator(data=data).build_skeleton(
            ci_test=_fake_ci_temporal,
            expert_knowledge=expert,
            variant="stable",
            show_progress=False,
        )

    assert skel.has_edge("A", "B"), (
        f"{estimator_class.__name__}: Edge A-B was incorrectly removed: "
        "temporal ordering should have filtered C from candidate separators "
        "(bug: min(u,u) instead of min(u,v))"
    )


def test_score():
    asia_model = load_model("bnlearn/asia")
    data = asia_model.simulate(n_samples=int(1e4), seed=42)
    est = PC(
        return_type="dag",
        variant="stable",
        max_cond_vars=4,
        show_progress=False,
    )

    with pytest.raises(NotFittedError):
        corr_score = est.score(X=data)

    est.fit(X=data)
    corr_score = est.score(X=data)
    shd = est.score(true_graph=asia_model)

    assert np.round(corr_score, 4) > 0.5
    assert shd, 2

    corr_score = est.score(X=data, metric=CorrelationScore(significance_level=0.01))
    shd = est.score(true_graph=asia_model, metric=SHD())

    assert np.round(corr_score, 4) > 0.5
    assert shd, 2

    structure_score = est.score(X=data, metric="structure_score")
    shd = est.score(true_graph=asia_model, metric="SHD")
    assert np.round(structure_score, 4) > -3e4
    assert shd, 2


def test_stable_variant_order_independence():

    rng = np.random.default_rng(seed=42)
    n = 2000

    data = pd.DataFrame(
        rng.integers(0, 3, size=(n, 12)),
        columns=[chr(65 + i) for i in range(12)],
    )
    data["M"] = (data["A"] + data["B"] + data["C"]) % 3
    data["N"] = (data["D"] + data["E"] + data["F"]) % 3
    data["O"] = (data["M"] + data["N"] + data["G"]) % 3

    skeletons = []
    for _ in range(5):
        cols = list(rng.permutation(data.columns))
        shuffled = data[cols]
        pc = PC(
            variant="stable",
            ci_test="chi_square",
            significance_level=0.05,
            show_progress=False,
        )
        pc.fit(shuffled)
        edges = frozenset(frozenset((u, v)) for u, v in pc.skeleton_.edges())
        skeletons.append(edges)

    assert all(s == skeletons[0] for s in skeletons)
