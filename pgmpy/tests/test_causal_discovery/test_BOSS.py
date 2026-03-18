import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import BOSS


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [BOSS(return_type="dag")],
    expected_failed_checks=expected_failed_checks,
)
def test_boss_compatibility(estimator, check):
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
def titanic_data_categorical(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")


class TestBOSSCore:
    """Tests for core BOSS functionality."""

    def test_estimate_rand(self, rand_data):
        est = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est.fit(rand_data)
        assert set(est.causal_graph_.nodes()) == set(["A", "B", "C"])
        assert list(est.causal_graph_.edges()) == [("B", "C")] or list(
            est.causal_graph_.edges()
        ) == [("C", "B")]

    def test_estimate_titanic(self, titanic_data_categorical):
        est = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est.fit(titanic_data_categorical)
        assert len(est.causal_graph_.edges()) > 0

    def test_return_type_pdag(self, rand_data):
        est = BOSS(scoring_method="k2", return_type="pdag", random_state=42)
        est.fit(rand_data)
        assert est.causal_graph_ is not None
        assert est.adjacency_matrix_ is not None

    def test_return_type_dag(self, rand_data):
        est = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est.fit(rand_data)
        assert est.causal_graph_ is not None
        assert est.adjacency_matrix_ is not None

    def test_adjacency_matrix(self, rand_data):
        est = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est.fit(rand_data)
        assert est.adjacency_matrix_ is not None
        assert est.adjacency_matrix_.shape[0] == len(rand_data.columns)
        assert est.adjacency_matrix_.shape[1] == len(rand_data.columns)

    def test_feature_names(self, rand_data):
        est = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est.fit(rand_data)
        assert hasattr(est, "n_features_in_")
        assert hasattr(est, "feature_names_in_")

    def test_determinism(self, rand_data):
        est1 = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est2 = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est1.fit(rand_data)
        est2.fit(rand_data)
        assert set(est1.causal_graph_.edges()) == set(est2.causal_graph_.edges())

    def test_result_is_dag(self, rand_data):
        est = BOSS(scoring_method="k2", return_type="dag", random_state=42)
        est.fit(rand_data)
        assert nx.is_directed_acyclic_graph(est.causal_graph_)


class TestBOSSScoringMethods:
    """Tests for different scoring methods."""

    @pytest.mark.parametrize("scoring_method", ["k2", "bdeu", "bds", "bic-d", "aic-d"])
    def test_discrete_scores(self, rand_data, scoring_method):
        est = BOSS(scoring_method=scoring_method, return_type="dag", random_state=42)
        est.fit(rand_data)

    @pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
    def test_gaussian_scores(self, scoring_method):
        data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0
        )
        est = BOSS(scoring_method=scoring_method, return_type="dag", random_state=42)
        est.fit(data)
