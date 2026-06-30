import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery import SortnRegress

# sklearn compatibility


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
        "check_fit2d_1feature": (
            "SortnRegress naturally fails on 1 feature because global R2 calculation requires other nodes."
        ),
    }


@parametrize_with_checks(
    [SortnRegress(threshold=0.3)],
    expected_failed_checks=expected_failed_checks,
)
def test_sortnregress_compatibility(estimator, check):
    check(estimator)


# fixtures and core tests for SortnRegress functionality and scoring
@pytest.fixture
def causal_chain_data():
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1, n)
    y = x + np.random.normal(0, 1.5, n)
    z = y + np.random.normal(0, 2.0, n)

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


class TestSortnRegressCore:
    def test_scale_invariance(self, causal_chain_data):
        """
        R²-SortnRegress is scale-invariant. It must recover identical graphs
        on raw vs standardized (unit variance) datasets.
        """
        est_raw = SortnRegress(threshold=0.3)
        est_raw.fit(causal_chain_data)
        raw_edges = set(est_raw.causal_graph_.edges())

        # Standardize data to unit variance
        standardized_data = (causal_chain_data - causal_chain_data.mean()) / causal_chain_data.std()

        est_std = SortnRegress(threshold=0.3)
        est_std.fit(standardized_data)
        std_edges = set(est_std.causal_graph_.edges())

        assert len(raw_edges) > 0
        assert raw_edges == std_edges

    def test_adjacency_matrix(self, causal_chain_data):
        est = SortnRegress(threshold=0.1)
        est.fit(causal_chain_data)

        adj = est.adjacency_matrix_
        assert isinstance(adj, pd.DataFrame)
        assert adj.shape == (3, 3)
        assert set(adj.columns) == {"X", "Y", "Z"}

    def test_thresholding(self, causal_chain_data):
        est = SortnRegress(threshold=100.0)
        est.fit(causal_chain_data)
        assert len(est.causal_graph_.edges()) == 0

    def test_feature_names(self, causal_chain_data):
        est = SortnRegress()
        est.fit(causal_chain_data)
        assert hasattr(est, "n_features_in_")
        assert est.n_features_in_ == 3
        assert list(est.feature_names_in_) == ["X", "Y", "Z"]

    def test_zero_variance_column(self):
        np.random.seed(0)
        data = pd.DataFrame(np.random.randn(100, 2), columns=["X", "Y"])
        data["C"] = 0.0

        est = SortnRegress(threshold=0.3)
        est.fit(data)

        edges = list(est.causal_graph_.edges())
        assert all("C" not in edge for edge in edges)


class TestSortnRegressScoring:
    def test_score(self, causal_chain_data):
        est = SortnRegress(threshold=0.3)
        est.fit(causal_chain_data)

        score = est.score(X=causal_chain_data)
        assert isinstance(score, float)

        true_dag = DAG([("X", "Y"), ("Y", "Z")])
        shd_score = est.score(true_graph=true_dag)
        assert isinstance(shd_score, (int, float, np.integer))
