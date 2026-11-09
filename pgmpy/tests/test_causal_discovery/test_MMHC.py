"""
Tests for the sklearn-compatible MMHC class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG, UndirectedGraph
from pgmpy.causal_discovery import MMHC, ExpertKnowledge
from pgmpy.ci_tests import ChiSquare, GeneralizedCov
from pgmpy.estimators import MmhcEstimator
from pgmpy.example_models import load_model
from pgmpy.structure_score import BDeu


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [MMHC(return_type="dag", max_iter=100, show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
def test_mmhc_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def cancer_data():
    return load_model("bnlearn/cancer").simulate(n_samples=300, seed=42)[
        ["Pollution", "Smoker", "Cancer", "Xray", "Dyspnoea"]
    ]


@pytest.fixture
def collider_data():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.integers(0, 2, size=(5000, 5)), columns=list("ABCDE"))
    data["F"] = data["A"] + data["B"] + data["C"]
    return data


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(42)
    a = rng.normal(size=500)
    b = a + rng.normal(size=500)
    return pd.DataFrame({"A": a, "B": b, "C": b + rng.normal(size=500)})


@pytest.mark.parametrize(("return_type", "graph_type"), [("pdag", PDAG), ("cpdag", PDAG), ("dag", DAG)])
def test_fit_sets_graph_attributes(return_type, graph_type, cancer_data):
    estimator = MMHC(return_type=return_type, show_progress=False, max_iter=100).fit(cancer_data)
    columns = list(cancer_data.columns)

    assert isinstance(estimator.causal_graph_, graph_type)
    assert set(estimator.causal_graph_.nodes()) == set(columns)
    assert estimator.n_features_in_ == len(columns)
    assert list(estimator.feature_names_in_) == columns
    assert isinstance(estimator.adjacency_matrix_, pd.DataFrame)
    assert estimator.adjacency_matrix_.shape == (len(columns), len(columns))
    assert set(estimator.adjacency_matrix_.index) == set(estimator.adjacency_matrix_.columns) == set(columns)

    assert isinstance(estimator.skeleton_, UndirectedGraph)
    graph_edges = {frozenset(edge) for edge in estimator.causal_graph_.edges()}
    assert graph_edges <= {frozenset(edge) for edge in estimator.skeleton_.edges()}
    assert graph_edges == {frozenset(("Cancer", "Smoker"))}


def test_dag_is_acyclic_and_scorable_after_fit(cancer_data):
    estimator = MMHC(return_type="dag", show_progress=False, max_iter=100)
    with pytest.raises(NotFittedError):
        estimator.score(X=cancer_data)

    estimator.fit(cancer_data)
    assert nx.is_directed_acyclic_graph(estimator.causal_graph_)
    assert isinstance(estimator.score(X=cancer_data), float)


def test_refit_on_numpy_input_with_constant_column(cancer_data):
    estimator = MMHC(show_progress=False, max_iter=100).fit(cancer_data)
    with pytest.warns(UserWarning, match="constant"):
        estimator.fit(cancer_data.iloc[:, :2].assign(constant="c").to_numpy())

    names = ["x0", "x1", "x2"]
    assert estimator.n_features_in_ == 3
    assert list(estimator.feature_names_in_) == names
    assert set(estimator.causal_graph_.nodes()) == set(names)
    assert estimator.adjacency_matrix_.shape == (3, 3)
    assert estimator.causal_graph_.degree("x2") == 0

    with pytest.raises((AttributeError, TypeError, ValueError)):
        estimator.fit([[0, 1], [1, 0]])


@pytest.mark.parametrize("data_fixture", ["cancer_data", "gaussian_data"])
def test_deprecated_estimator_keeps_chi_square_and_bdeu_defaults(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    with pytest.warns(FutureWarning, match="pgmpy\\.causal_discovery\\.MMHC"):
        legacy = MmhcEstimator(data)

    expected = MMHC(
        ci_test="chi_square",
        scoring_method=BDeu(data, equivalent_sample_size=10),
        return_type="dag",
        show_progress=False,
    ).fit(data)
    assert set(legacy.estimate().edges()) == set(expected.causal_graph_.edges())
    assert set(legacy.mmpc().edges()) == set(expected.skeleton_.edges())


@pytest.mark.parametrize(
    ("ci_test_name", "max_cond_vars", "largest_conditioning_set"),
    [("chi_square", None, None), ("generalized_cov", None, None), ("callable", None, 2), ("callable", 1, 1)],
)
def test_skeleton_recovers_collider(collider_data, ci_test_name, max_cond_vars, largest_conditioning_set):
    chi_square = ChiSquare(collider_data)
    conditioning_sets = []

    def recording_chi_square(X, Y, Z, significance_level):
        conditioning_sets.append(Z)
        return chi_square(X, Y, Z, significance_level=significance_level)

    ci_test = {
        "chi_square": chi_square,
        "generalized_cov": GeneralizedCov(collider_data),
        "callable": recording_chi_square,
    }[ci_test_name]
    skeleton = MMHC(ci_test=ci_test, max_cond_vars=max_cond_vars, show_progress=False).fit(collider_data).skeleton_

    assert {frozenset(edge) for edge in skeleton.edges()} == {frozenset("AF"), frozenset("BF"), frozenset("CF")}
    if largest_conditioning_set is not None:
        assert max(len(Z) for Z in conditioning_sets) == largest_conditioning_set


@pytest.mark.parametrize("reverse_columns", [False, True])
def test_skeleton_recovers_sachs_in_any_column_order(reverse_columns):
    model = load_model("bnlearn/sachs")
    data = model.simulate(n_samples=2000, seed=42, show_progress=False)
    columns = sorted(data.columns, reverse=reverse_columns)

    skeleton = MMHC(show_progress=False).fit(data[columns]).skeleton_
    assert {frozenset(edge) for edge in skeleton.edges()} == {frozenset(edge) for edge in model.edges()}


@pytest.mark.parametrize("use_tiers", [False, True])
@pytest.mark.parametrize("reverse_columns", [False, True])
@pytest.mark.parametrize("graph", ["cancer", "asia", "sachs", "seven_nodes"])
def test_skeleton_is_exact_with_oracle_ci_test(graph, reverse_columns, use_tiers):
    if graph == "seven_nodes":
        dag = DAG([("X1", "X0"), ("X1", "X3"), ("X1", "X6"), ("X3", "X2"), ("X5", "X6"), ("X6", "X0"), ("X6", "X3")])
        dag.add_node("X4")
    else:
        dag = DAG(load_model(f"bnlearn/{graph}").edges())

    nodes = sorted(dag.nodes(), reverse=reverse_columns)
    data = pd.DataFrame(np.random.default_rng(42).integers(0, 2, size=(100, len(nodes))), columns=nodes)
    tiers = [sorted(generation) for generation in nx.topological_generations(dag)]
    tier = {node: index for index, generation in enumerate(tiers) for node in generation}
    tests_run = []

    def d_separation_oracle(X, Y, Z, significance_level):
        tests_run.append((X, Y, Z))
        return not dag.is_dconnected(X, Y, observed=list(Z))

    expert = ExpertKnowledge(temporal_order=tiers) if use_tiers else None
    skeleton = MMHC(ci_test=d_separation_oracle, expert_knowledge=expert, show_progress=False).fit(data).skeleton_

    assert {frozenset(edge) for edge in skeleton.edges()} == {frozenset(edge) for edge in dag.edges()}
    if use_tiers:
        assert all(tier[z] <= max(tier[X], tier[Y]) for X, Y, Z in tests_run for z in Z)


@pytest.mark.parametrize(("data_fixture", "data_type"), [("cancer_data", "discrete"), ("gaussian_data", "continuous")])
def test_defaults_and_tags_follow_data_type(data_fixture, data_type, request):
    data = request.getfixturevalue(data_fixture)
    estimator = MMHC(show_progress=False).fit(data)

    assert estimator.get_tags()["data_types"] == [data_type]
    assert data_type in estimator.ci_test_.get_tags()["data_types"]
    assert data_type in estimator.scoring_method_.get_tags()["data_types"]


def test_stricter_significance_level_is_no_denser(cancer_data):
    permissive = MMHC(significance_level=0.05, show_progress=False, max_iter=100).fit(cancer_data)
    strict = MMHC(significance_level=0.001, show_progress=False, max_iter=100).fit(cancer_data)

    assert strict.skeleton_.number_of_edges() <= permissive.skeleton_.number_of_edges()


def test_expert_knowledge_is_applied_without_mutation(collider_data):
    data_before = collider_data.copy(deep=True)
    expert = ExpertKnowledge(forbidden_edges=[("A", "F"), ("F", "A")], required_edges=[("D", "E")])
    expert_before = expert.get_params(deep=True)

    estimator = MMHC(expert_knowledge=expert, return_type="dag", show_progress=False, max_iter=100).fit(collider_data)

    assert {frozenset(edge) for edge in estimator.skeleton_.edges()} == {
        frozenset("BF"),
        frozenset("CF"),
        frozenset("DE"),
    }
    assert ("D", "E") in estimator.causal_graph_.edges()
    pd.testing.assert_frame_equal(collider_data, data_before)
    assert expert.get_params(deep=True) == expert_before
    assert not hasattr(expert, "forbidden_edges_")
    assert not hasattr(expert, "required_edges_")


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"return_type": "invalid"}, "return_type"),
        ({"max_cond_vars": -1}, "max_cond_vars"),
        ({"ci_test": "not_a_ci_test"}, "Unknown CI test"),
        ({"scoring_method": "not_a_score"}, "Unknown scoring method"),
    ],
)
def test_invalid_parameters_raise(params, match, cancer_data):
    with pytest.raises(ValueError, match=match):
        MMHC(show_progress=False, **params).fit(cancer_data)
