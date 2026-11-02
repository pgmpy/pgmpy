import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import SP
from pgmpy.example_models import load_model


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [SP(max_iter=2)],
    expected_failed_checks=expected_failed_checks,
)
def test_sp_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def cancer_data():
    model = load_model("bnlearn/cancer")
    return model.simulate(n_samples=1000, seed=42)


class TestSP:
    def test_chain_recovery_and_attributes(self, cancer_data):
        est = SP(ci_test="g_sq", return_type="dag", seed=42)
        est.fit(cancer_data)
        assert isinstance(est.causal_graph_, DAG)

        assert set(est.causal_graph_.edges()) == {("Xray", "Cancer")}
        assert est.n_features_in_ == cancer_data.shape[1]
        assert list(est.feature_names_in_) == list(cancer_data.columns)

        adj = est.adjacency_matrix_
        assert sorted(adj.index) == sorted(cancer_data.columns)
        assert sorted(adj.columns) == sorted(cancer_data.columns)
        assert np.all(np.diag(adj) == 0)

    def test_returns_pdag(self, cancer_data):
        est = SP(ci_test="g_sq", return_type="pdag")
        est.fit(cancer_data)
        assert isinstance(est.causal_graph_, PDAG)

    def test_seed_and_max_iter(self, cancer_data):
        est1 = SP(ci_test="g_sq", max_iter=10, seed=42)
        est2 = SP(ci_test="g_sq", max_iter=10, seed=42)
        est1.fit(cancer_data)
        est2.fit(cancer_data)
        assert set(est1.causal_graph_.edges()) == set(est2.causal_graph_.edges())
        assert est1.optimal_permutations_ == est2.optimal_permutations_
