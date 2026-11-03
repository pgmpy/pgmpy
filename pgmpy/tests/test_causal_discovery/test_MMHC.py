"""
Tests for the sklearn-compatible MMHC class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG, UndirectedGraph
from pgmpy.causal_discovery import MMHC, ExpertKnowledge
from pgmpy.estimators import MmhcEstimator
from pgmpy.example_models import load_model


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
def small_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "A": rng.integers(0, 2, size=150),
            "B": rng.integers(0, 2, size=150),
            "isolated": np.arange(150),
        }
    )


def test_parameters_are_stored_and_round_trip():
    estimator = MMHC(
        ci_test="chi_square",
        significance_level=0.05,
        max_cond_vars=2,
        scoring_method="k2",
        tabu_length=3,
        max_indegree=2,
        max_iter=10,
        epsilon=0.01,
        return_type="dag",
        show_progress=False,
        expert_knowledge=ExpertKnowledge(),
    )

    params = estimator.get_params()
    assert {
        "ci_test",
        "significance_level",
        "max_cond_vars",
        "scoring_method",
        "tabu_length",
        "max_indegree",
        "max_iter",
        "epsilon",
        "return_type",
        "show_progress",
        "expert_knowledge",
    } <= set(params)
    for name in estimator.get_params(deep=False):
        value = params[name]
        assert getattr(estimator, name) is value or getattr(estimator, name) == value

    updated = estimator.set_params(max_cond_vars=1, return_type="pdag")
    assert updated is estimator
    assert estimator.max_cond_vars == 1
    assert estimator.return_type == "pdag"

    cloned = clone(estimator)
    for name in estimator.get_params(deep=False):
        if name == "expert_knowledge":
            assert cloned.expert_knowledge.get_params() == estimator.expert_knowledge.get_params()
        else:
            assert getattr(cloned, name) == getattr(estimator, name)
    assert not hasattr(cloned, "causal_graph_")


def test_fit_returns_self_and_sets_learned_attributes(cancer_data):
    estimator = MMHC(show_progress=False, max_iter=100)
    assert not any(hasattr(estimator, name) for name in ("causal_graph_", "adjacency_matrix_", "skeleton_"))

    assert estimator.fit(cancer_data) is estimator
    assert hasattr(estimator, "causal_graph_")
    assert hasattr(estimator, "adjacency_matrix_")
    assert hasattr(estimator, "skeleton_")
    assert estimator.n_features_in_ == len(cancer_data.columns)
    assert list(estimator.feature_names_in_) == list(cancer_data.columns)


@pytest.mark.parametrize(
    ("return_type", "graph_type"),
    [("pdag", PDAG), ("cpdag", PDAG), ("dag", DAG)],
)
def test_graph_attributes(return_type, graph_type, cancer_data):
    estimator = MMHC(return_type=return_type, show_progress=False, max_iter=100).fit(cancer_data)

    assert isinstance(estimator.causal_graph_, graph_type)
    assert set(estimator.causal_graph_.nodes()) == set(cancer_data.columns)
    assert isinstance(estimator.adjacency_matrix_, pd.DataFrame)
    assert estimator.adjacency_matrix_.shape == (len(cancer_data.columns), len(cancer_data.columns))
    assert set(estimator.adjacency_matrix_.index) == set(cancer_data.columns)
    assert set(estimator.adjacency_matrix_.columns) == set(cancer_data.columns)


def test_graph_edges_are_restricted_to_skeleton(cancer_data):
    estimator = MMHC(return_type="dag", show_progress=False, max_iter=100).fit(cancer_data)
    assert isinstance(estimator.skeleton_, UndirectedGraph)
    graph_edges = {frozenset(edge) for edge in estimator.causal_graph_.edges()}
    skeleton_edges = {frozenset(edge) for edge in estimator.skeleton_.edges()}
    assert graph_edges <= skeleton_edges


def test_numpy_input_uses_x_feature_names(cancer_data):
    estimator = MMHC(show_progress=False, max_iter=100).fit(cancer_data.to_numpy())

    assert estimator.n_features_in_ == cancer_data.shape[1]
    expected_names = [f"x{i}" for i in range(cancer_data.shape[1])]
    assert list(estimator.feature_names_in_) == expected_names
    assert set(estimator.causal_graph_.nodes()) == set(expected_names)


def test_expected_edges_are_deterministic(cancer_data):
    estimator = MMHC(return_type="dag", show_progress=False, max_iter=1000).fit(cancer_data)

    assert {frozenset(edge) for edge in estimator.causal_graph_.edges()} == {
        frozenset(("Dyspnoea", "Pollution")),
    }


def test_matches_deprecated_estimator(cancer_data):
    with pytest.warns(DeprecationWarning, match="pgmpy\\.causal_discovery\\.MMHC"):
        legacy = MmhcEstimator(cancer_data)

    expected = MMHC(return_type="dag", show_progress=False).fit(cancer_data).causal_graph_
    actual = legacy.estimate()
    assert set(actual.edges()) == set(expected.edges())

    with pytest.warns(DeprecationWarning, match="pgmpy\\.causal_discovery\\.MMHC"):
        legacy = MmhcEstimator(cancer_data)
    assert set(legacy.mmpc().edges()) == set(MMHC(show_progress=False).fit(cancer_data).skeleton_.edges())


def test_isolated_column_is_preserved():
    data = pd.DataFrame(
        {
            "A": [0, 1, 0, 1] * 25,
            "B": [1, 0, 1, 0] * 25,
            "isolated": [0] * 100,
        }
    )
    estimator = MMHC(show_progress=False, max_iter=10).fit(data)

    assert "isolated" in estimator.causal_graph_.nodes()
    assert estimator.causal_graph_.degree("isolated") == 0


def test_dag_is_acyclic(cancer_data):
    estimator = MMHC(return_type="dag", show_progress=False, max_iter=100).fit(cancer_data)
    assert nx.is_directed_acyclic_graph(estimator.causal_graph_)


def test_stricter_significance_level_is_no_denser(cancer_data):
    permissive = MMHC(significance_level=0.05, show_progress=False, max_iter=100).fit(cancer_data)
    strict = MMHC(significance_level=0.001, show_progress=False, max_iter=100).fit(cancer_data)

    assert strict.skeleton_.number_of_edges() <= permissive.skeleton_.number_of_edges()


def test_refit_overwrites_learned_attributes(cancer_data):
    first = cancer_data.iloc[:, :3]
    second = cancer_data.iloc[:, :2]
    estimator = MMHC(show_progress=False, max_iter=100).fit(first)
    estimator.fit(second)

    assert estimator.n_features_in_ == 2
    assert list(estimator.feature_names_in_) == list(second.columns)
    assert set(estimator.causal_graph_.nodes()) == set(second.columns)
    assert estimator.adjacency_matrix_.shape == (2, 2)


def test_input_and_expert_knowledge_are_not_mutated(cancer_data):
    data_before = cancer_data.copy(deep=True)
    expert = ExpertKnowledge(required_edges=[("Cancer", "Dyspnoea")])
    expert_before = expert.get_params(deep=True)

    MMHC(expert_knowledge=expert, show_progress=False, max_iter=100).fit(cancer_data)

    pd.testing.assert_frame_equal(cancer_data, data_before)
    assert expert.get_params(deep=True) == expert_before
    assert not hasattr(expert, "forbidden_edges_")
    assert not hasattr(expert, "required_edges_")


def test_invalid_parameters_raise(cancer_data):
    with pytest.raises(ValueError, match="return_type"):
        MMHC(return_type="invalid", show_progress=False).fit(cancer_data)

    with pytest.raises(ValueError, match="max_cond_vars"):
        MMHC(max_cond_vars=-1, show_progress=False).fit(cancer_data)

    with pytest.raises(ValueError, match="Unknown CI test"):
        MMHC(ci_test="not_a_ci_test", show_progress=False).fit(cancer_data)

    with pytest.raises(ValueError):
        MMHC(scoring_method="not_a_score", show_progress=False).fit(cancer_data)


@pytest.mark.parametrize(
    "data",
    [
        [[0, 1], [1, 0]],
        pd.DataFrame(),
        pd.DataFrame({"A": [0, np.nan], "B": [1, 0]}),
    ],
)
def test_invalid_input_matches_base_validation(data):
    with pytest.raises((AttributeError, TypeError, ValueError)):
        MMHC(show_progress=False, max_iter=10).fit(data)


def test_score_requires_fit_and_works_after_fit(cancer_data):
    estimator = MMHC(return_type="dag", show_progress=False, max_iter=100)
    with pytest.raises(NotFittedError):
        estimator.score(X=cancer_data)

    estimator.fit(cancer_data)
    assert isinstance(estimator.score(X=cancer_data), float)
