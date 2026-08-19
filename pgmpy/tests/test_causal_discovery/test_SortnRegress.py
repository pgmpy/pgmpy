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
    [SortnRegress()],
    expected_failed_checks=expected_failed_checks,
)
def test_sortnregress_compatibility(estimator, check):
    check(estimator)


# fixtures and core tests for SortnRegress functionality and scoring
@pytest.fixture
def causal_chain_data():
    rng = np.random.default_rng(seed=42)
    n = 1000
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, 1.5, n)
    z = y + rng.normal(0, 2.0, n)

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


class TestSortnRegressCore:
    def test_scale_invariance(self, causal_chain_data):
        """
        R²-SortnRegress is scale-invariant. It must recover identical graphs
        on raw vs standardized (unit variance) datasets.
        """
        rng = np.random.default_rng(seed=7)
        est_raw = SortnRegress().fit(causal_chain_data)
        scaled = causal_chain_data * rng.uniform(0.05, 20, causal_chain_data.shape[1])
        est_scaled = SortnRegress().fit(scaled)
        assert len(est_raw.causal_graph_.edges()) > 0
        assert set(est_raw.causal_graph_.edges()) == set(est_scaled.causal_graph_.edges())

    def test_adjacency_matrix(self, causal_chain_data):
        est = SortnRegress()
        est.fit(causal_chain_data)

        adj = est.adjacency_matrix_
        assert isinstance(adj, pd.DataFrame)
        assert adj.shape == (3, 3)
        assert set(adj.columns) == {"X", "Y", "Z"}

    def test_feature_names(self, causal_chain_data):
        est = SortnRegress()
        est.fit(causal_chain_data)
        assert hasattr(est, "n_features_in_")
        assert est.n_features_in_ == 3
        assert list(est.feature_names_in_) == ["X", "Y", "Z"]

    def test_zero_variance_column(self):
        rng = np.random.default_rng(seed=0)
        data = pd.DataFrame(rng.standard_normal((100, 2)), columns=["X", "Y"])
        data["C"] = 0.0

        est = SortnRegress()
        with pytest.raises(ValueError, match="zero variance"):
            est.fit(data)


class TestSortnRegressScoring:
    def test_score(self, causal_chain_data):
        est = SortnRegress()
        est.fit(causal_chain_data)

        score = est.score(X=causal_chain_data)
        assert isinstance(score, float)

        true_dag = DAG([("X", "Y"), ("Y", "Z")])
        shd_score = est.score(true_graph=true_dag)
        assert isinstance(shd_score, (int, float, np.integer))


class TestSortnRegressvariant:
    def test_default_variant_is_r2(self, causal_chain_data):
        est = SortnRegress()  # default variant is r2
        assert est.variant == "r2"
        est.fit(causal_chain_data)
        assert len(est.causal_graph_.edges()) > 0

    def test_varsortability_variant_fits(self, causal_chain_data):
        est = SortnRegress(variant="varsortability")
        est.fit(causal_chain_data)
        assert len(est.causal_graph_.edges()) > 0

    def test_invalid_variant_raises(self, causal_chain_data):
        est = SortnRegress(variant="not_a_real_variant")
        with pytest.raises(ValueError, match="variant must be one of"):
            est.fit(causal_chain_data)

    def test_variance_variant_is_scale_sensitive(self, causal_chain_data):
        std = (causal_chain_data - causal_chain_data.mean()) / causal_chain_data.std()
        raw = SortnRegress(variant="varsortability").fit(causal_chain_data)
        stdz = SortnRegress(variant="varsortability").fit(std)
        assert set(raw.causal_graph_.edges()) != set(stdz.causal_graph_.edges())
