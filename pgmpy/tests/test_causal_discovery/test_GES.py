"""
Tests for the sklearn-compatible GES class in pgmpy.causal_discovery.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import GES
from pgmpy.estimators import ExpertKnowledge


def make_estimator():
    return GES()


@parametrize_with_checks([make_estimator()])
def test_ges_sklearn_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def rand_data():
    data = pd.DataFrame(
        np.random.randint(0, 5, size=(int(1e4), 2)),
        columns=list("AB"),
        dtype="category",
    )
    data["C"] = data["B"]
    return data


@pytest.fixture
def titanic_data():
    return pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")


@pytest.fixture
def titanic_data1(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass", "Age", "Embarked"]].dropna()


@pytest.fixture
def titanic_data2(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")


def test_estimate_rand(rand_data):
    est = GES(scoring_method="k2", return_type="dag")
    est.fit(rand_data)
    assert set(est.causal_graph_.nodes()) == set(["A", "B", "C"])
    assert list(est.causal_graph_.edges()) == [("B", "C")] or list(
        est.causal_graph_.edges()
    ) == [("C", "B")]


def test_estimate_with_expert_knowledge(rand_data):
    expert_knowledge = ExpertKnowledge(required_edges=[("B", "C")])
    est = GES(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
    )
    est.fit(rand_data)
    assert ("B", "C") in list(est.causal_graph_.edges())


def test_estimate_titanic(titanic_data2):
    est = GES(scoring_method="k2", return_type="dag")
    est.fit(titanic_data2)
    assert len(est.causal_graph_.edges()) > 0


def test_estimate_with_forbidden_edges(titanic_data2):
    expert_knowledge = ExpertKnowledge(
        forbidden_edges=[("Sex", "Survived"), ("Survived", "Sex")]
    )
    est = GES(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
    )
    est.fit(titanic_data2)
    assert ("Sex", "Survived") not in est.causal_graph_.edges()
    assert ("Survived", "Sex") not in est.causal_graph_.edges()


def test_return_type_pdag(rand_data):
    est = GES(scoring_method="k2", return_type="pdag")
    est.fit(rand_data)
    assert est.causal_graph_ is not None
    assert est.adjacency_matrix_ is not None


def test_return_type_dag(rand_data):
    est = GES(scoring_method="k2", return_type="dag")
    est.fit(rand_data)
    assert est.causal_graph_ is not None
    assert est.adjacency_matrix_ is not None


@pytest.mark.parametrize("scoring_method", ["k2", "bdeu", "bds", "bic-d", "aic-d"])
def test_estimate_discrete(rand_data, scoring_method):
    est = GES(scoring_method=scoring_method, return_type="dag")
    est.fit(rand_data)


@pytest.mark.parametrize("scoring_method", ["ll-cg", "aic-cg", "bic-cg"])
def test_estimate_mixed(titanic_data1, scoring_method):
    est = GES(scoring_method=scoring_method, return_type="dag")
    est.fit(titanic_data1)


@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_estimate_gaussian(scoring_method):
    data = pd.read_csv(
        "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0
    )
    est = GES(scoring_method=scoring_method, return_type="dag")
    est.fit(data)


def test_adjacency_matrix(rand_data):
    est = GES(scoring_method="k2", return_type="dag")
    est.fit(rand_data)
    assert est.adjacency_matrix_ is not None
    assert est.adjacency_matrix_.shape[0] == len(rand_data.columns)
    assert est.adjacency_matrix_.shape[1] == len(rand_data.columns)


def test_feature_names(rand_data):
    est = GES(scoring_method="k2", return_type="dag")
    est.fit(rand_data)
    assert hasattr(est, "n_features_in_")
    assert hasattr(est, "feature_names_in_")


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
    est = GES(scoring_method="k2", expert_knowledge=expert_knowledge, return_type="dag")
    est.fit(adult_data)
    for edge in est.causal_graph_.edges():
        assert edge in search_space


def test_temporal_order(titanic_data2):
    expert_knowledge = ExpertKnowledge(temporal_order=[["Pclass", "Sex"], ["Survived"]])
    est = GES(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
    )
    est.fit(titanic_data2)
    expected_edges = {("Sex", "Survived"), ("Sex", "Pclass"), ("Pclass", "Survived")}
    assert set(est.causal_graph_.edges()) == expected_edges
