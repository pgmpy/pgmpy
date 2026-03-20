import logging

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.estimators import PC, ExpertKnowledge
from pgmpy.example_models import load_model
from pgmpy.independencies import Independencies
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)


@pytest.fixture
def estimator():
    np.random.seed(42)
    fake_data = pd.DataFrame(np.random.random((1000, 4)), columns=["A", "B", "C", "D"])
    return PC(fake_data)


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


@pytest.mark.parametrize("variant", ["orig", "stable"])
def test_build_skeleton_fake_ci(estimator, variant):
    skel, _ = estimator.build_skeleton(ci_test=fake_ci_t, variant=variant)
    expected_edges = {("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert (u, v) in expected_edges or (v, u) in expected_edges


@pytest.mark.parametrize("variant", ["orig", "stable"])
def test_build_skeleton_max_cond_vars_0(estimator, variant):
    skel, _ = estimator.build_skeleton(ci_test=fake_ci_t, variant=variant, max_cond_vars=0)
    expected_edges = {("A", "B"), ("A", "C"), ("A", "D")}
    for u, v in skel.edges():
        assert (u, v) in expected_edges or (v, u) in expected_edges


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_from_ind(variant):
    ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
    ind = ind.closure()
    estimator = PC(independencies=ind)
    skel, sep_sets = estimator.estimate(
        variant=variant,
        ci_test="independence_match",
        return_type="skeleton",
        n_jobs=2,
        show_progress=False,
    )

    expected_edges = {("A", "D"), ("B", "D"), ("C", "D")}
    expected_sepsets = {
        frozenset(("A", "C")): tuple(),
        frozenset(("A", "B")): tuple(),
        frozenset(("C", "B")): tuple(),
    }

    for u, v in skel.edges():
        assert (u, v) in expected_edges or (v, u) in expected_edges
    assert sep_sets == expected_sepsets


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_from_model_ind(variant):
    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
    estimator = PC(independencies=model.get_independencies())
    skel, sep_sets = estimator.estimate(
        variant=variant,
        ci_test="independence_match",
        return_type="skeleton",
        n_jobs=2,
        show_progress=False,
    )

    expected_edges = model.edges()
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
    for u, v in skel.edges():
        assert (u, v) in expected_edges or ((v, u) in expected_edges)
    assert sep_sets == expected_sepsets1 or sep_sets == expected_sepsets2


@pytest.mark.parametrize(
    ("skel", "sep_sets", "expected_edges"),
    [
        (
            nx.Graph([("A", "D"), ("A", "C"), ("B", "C")]),
            {
                frozenset({"D", "C"}): ("A",),
                frozenset({"A", "B"}): tuple(),
                frozenset({"D", "B"}): ("A",),
            },
            {("B", "C"), ("A", "D"), ("A", "C"), ("D", "A")},
        ),
        (
            nx.Graph([("A", "B"), ("A", "C")]),
            {frozenset({"B", "C"}): ()},
            {("B", "A"), ("C", "A")},
        ),
        (
            nx.Graph([("A", "B"), ("A", "C")]),
            {frozenset({"B", "C"}): ("A",)},
            {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")},
        ),
        (
            nx.Graph([("A", "C"), ("B", "C"), ("C", "D")]),
            {
                frozenset({"A", "B"}): tuple(),
                frozenset({"A", "D"}): ("C",),
                frozenset({"B", "D"}): ("C",),
            },
            {("A", "C"), ("B", "C"), ("C", "D")},
        ),
        (
            nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")]),
            {frozenset({"A", "D"}): tuple(), frozenset({"C", "D"}): ("A", "B")},
            {("A", "B"), ("B", "C"), ("A", "C"), ("D", "B")},
        ),
        (
            nx.Graph([("A", "B"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")]),
            {frozenset({"A", "C"}): ("B",)},
            {
                ("A", "B"),
                ("B", "A"),
                ("B", "C"),
                ("C", "B"),
                ("A", "D"),
                ("B", "D"),
                ("C", "D"),
            },
        ),
    ],
)
def test_skeleton_to_pdag(skel, sep_sets, expected_edges):
    pdag = PC.orient_colliders(skeleton=skel, separating_sets=sep_sets)
    pdag = pdag.apply_meeks_rules(apply_r4=False)
    assert set(pdag.edges()) == expected_edges


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_estimate_dag(variant):
    ind = Independencies(["B", "C"], ["A", ["B", "C"], "D"])
    ind = ind.closure()
    estimator = PC(independencies=ind)
    model = estimator.estimate(
        variant=variant,
        ci_test="independence_match",
        return_type="dag",
        n_jobs=2,
        show_progress=False,
    )
    assert model.edges() == {("B", "D"), ("A", "D"), ("C", "D")}


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_estimate_dag_from_model(variant):
    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")])
    estimator = PC(independencies=model.get_independencies())
    estimated_model = estimator.estimate(
        variant=variant,
        ci_test="independence_match",
        return_type="dag",
        n_jobs=2,
        show_progress=False,
    )
    expected_edges_1 = set(model.edges())
    expected_edges_2 = {("B", "C"), ("A", "C"), ("C", "E"), ("D", "B")}
    assert set(estimated_model.edges()) == expected_edges_1 or set(estimated_model.edges()) == expected_edges_2


@pytest.fixture
def discrete_data():
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    return PC(data=data)


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_chi_square(variant):
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 2, size=(10000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    est = PC(data=data)
    skel, sep_sets = est.estimate(
        variant=variant,
        ci_test="chi_square",
        return_type="skeleton",
        significance_level=0.005,
        show_progress=False,
    )
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
    for u, v in skel.edges():
        assert (u, v) in expected_edges or (v, u) in expected_edges
    assert sep_sets == expected_sepsets


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_skeleton_fake_ci_discrete(variant):
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
    est = PC(data=fake_data)
    skel, sep_sets = est.estimate(
        variant=variant,
        ci_test=fake_ci,
        return_type="skeleton",
        show_progress=False,
    )
    expected_edges = {("X", "Z"), ("Y", "Z")}
    expected_sepsets = {frozenset(("X", "Y")): ("Z",)}
    for u, v in skel.edges():
        assert (u, v) in expected_edges or (v, u) in expected_edges
    assert sep_sets == expected_sepsets


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
@pytest.mark.parametrize("ci_test", ["g_sq", "log_likelihood", "modified_log_likelihood", "power_divergence"])
def test_build_skeleton_ci_tests(discrete_data, variant, ci_test):
    discrete_data.estimate(
        variant=variant,
        ci_test=ci_test,
        return_type="skeleton",
        significance_level=0.005,
        n_jobs=2,
        show_progress=False,
    )


@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
def test_build_dag_discrete(variant):
    np.random.seed(42)
    data = pd.DataFrame(np.random.randint(0, 3, size=(10000, 3)), columns=list("XYZ"))
    data["sum"] = data.sum(axis=1)
    est = PC(data=data)
    dag = est.estimate(
        variant=variant,
        ci_test="chi_square",
        return_type="dag",
        significance_level=0.001,
        n_jobs=2,
        show_progress=False,
    )

    assert set(dag.edges()) == {("Z", "sum"), ("X", "sum"), ("Y", "sum")}


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

    est = PC(adult_data)

    dag = est.estimate(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        enforce_expert_knowledge=True,
        show_progress=False,
    )
    for edge in dag.edges():
        assert edge in search_space


requires_xgboost = pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)


@requires_xgboost
@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
@pytest.mark.parametrize("ci_test", ["pearsonr", "pillai", "gcm"])
def test_build_skeleton_continuous(ci_test, variant):
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 5), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    est = PC(data=data)
    skel, sep_sets = est.estimate(
        variant=variant,
        ci_test=ci_test,
        return_type="skeleton",
        n_jobs=2,
        show_progress=False,
    )
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
        frozenset(("C", "B")): tuple(),
    }
    for u, v in skel.edges():
        assert (u, v) in expected_edges_stable or (v, u) in expected_edges_stable

    for key in sep_sets:
        assert sep_sets[key] == expected_sepsets[key]


@requires_xgboost
@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
@pytest.mark.parametrize("ci_test", ["pearsonr", "pillai", "gcm"])
def test_build_skeleton_continuous_fake_ci(ci_test, variant):
    def fake_ci(X, Y, Z=tuple(), **kwargs):
        if X == "X" and Y == "Y" and Z == ("Z",):
            return True
        elif X == "Y" and Y == "X" and Z == ("Z",):
            return True
        else:
            return False

    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
    est = PC(data=data)
    skel, sep_sets = est.estimate(
        variant=variant,
        ci_test=fake_ci,
        return_type="skeleton",
        n_jobs=2,
        show_progress=False,
    )
    expected_edges = {("X", "Z"), ("Y", "Z")}
    expected_sepsets = {frozenset(("X", "Y")): ("Z",)}

    for u, v in skel.edges():
        assert (u, v) in expected_edges or (v, u) in expected_edges
    assert sep_sets == expected_sepsets


@requires_xgboost
@pytest.mark.parametrize("variant", ["orig", "stable", "parallel"])
@pytest.mark.parametrize("ci_test", ["pearsonr", "pillai", "gcm"])
def test_build_dag_continuous(ci_test, variant):
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10000, 3), columns=list("XYZ"))
    data["sum"] = data.sum(axis=1)
    est = PC(data=data)
    dag = est.estimate(
        variant=variant,
        ci_test=ci_test,
        return_type="dag",
        n_jobs=2,
        show_progress=False,
    )

    assert set(dag.edges()) == {("Z", "sum"), ("X", "sum"), ("Y", "sum")}


def test_pc_alarm():
    alarm_model = load_model("bnlearn/alarm")
    data = BayesianModelSampling(alarm_model).forward_sample(size=int(1e4), seed=42)
    est = PC(data)
    est.estimate(variant="stable", max_cond_vars=5, n_jobs=2, show_progress=False)


def test_pc_asia(caplog):
    asia_model = load_model("bnlearn/asia")
    data = asia_model.simulate(n_samples=int(1e5), seed=42)
    est = PC(data)
    pgmpy_logger = logging.getLogger("pgmpy")
    pgmpy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="pgmpy"):
            est.estimate(
                variant="stable",
                max_cond_vars=4,
                expert_knowledge=ExpertKnowledge(required_edges=[("xray", "either")]),
                n_jobs=2,
                show_progress=False,
            )
    finally:
        pgmpy_logger.removeHandler(caplog.handler)
    assert (
        "Specified expert knowledge conflicts with learned structure. Ignoring edge xray->either from required edges"
    ) in caplog.text


def test_pc_asia_expert():
    asia_model = load_model("bnlearn/asia")
    data = asia_model.simulate(n_samples=int(1e5), seed=42)
    est = PC(data)
    pdag = est.estimate(
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
    edges = set(pdag.edges())
    directed = set(pdag.directed_edges)
    if ("lung", "either") in edges or ("either", "lung") in edges:
        assert ("lung", "either") in directed
    if ("tub", "either") in edges or ("either", "tub") in edges:
        assert ("tub", "either") in directed
    if ("bronc", "dysp") in pdag.edges() or ("dysp", "bronc") in pdag.edges():
        assert ("bronc", "dysp") in directed


def test_temporal_pc_cancer():
    cancer_model = load_model("bnlearn/cancer")
    data = cancer_model.simulate(n_samples=int(5e4), seed=42)
    est = PC(data)
    background = ExpertKnowledge(
        temporal_order=[["Pollution", "Smoker", "Cancer"], ["Dyspnoea", "Xray"]],
        max_cond_vars=4,
    )
    pdag = est.estimate(
        variant="stable",
        expert_knowledge=background,
        n_jobs=2,
        show_progress=False,
    )

    assert set(pdag.edges()) == {
        ("Cancer", "Xray"),
        ("Cancer", "Dyspnoea"),
        ("Smoker", "Cancer"),
        ("Pollution", "Cancer"),
    }


def test_temporal_pc_sachs():
    temporal_order = [
        ["PKC", "Plcg"],
        ["PKA", "Raf", "Jnk", "P38", "PIP3", "PIP2", "Mek", "Erk"],
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
    pdag = PC(df).estimate(ci_test="chi_square", expert_knowledge=expert)

    assert temporal_forbidden_edges.isdisjoint(set(pdag.edges()))
