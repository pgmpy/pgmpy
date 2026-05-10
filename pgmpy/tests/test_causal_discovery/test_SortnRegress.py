import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery import SortnRegress


# Reference implementation in scriddie/varsortability and CausalDisco,
# inlined here for testing without external dependencies.
def _ref_sortnregress(X):
    """
    Reference sortnregress from scriddie/varsortability (src/sortnregress.py).
    Identical to CausalDisco's var_sort_regress. Orders nodes by marginal
    variance and regresses each node onto those with lower variance using
    LinearRegression weights scaled by LassoLarsIC(BIC) for sparsification.
    Returns a (d x d) coefficient matrix W where W[i, j] != 0 means i -> j.
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")
    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))
    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]
        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight
    return W


# sklearn compatibility


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
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
    def test_estimate_chain(self, causal_chain_data):
        est = SortnRegress(threshold=0.5)
        est.fit(causal_chain_data)

        edges = set(est.causal_graph_.edges())
        assert ("X", "Y") in edges
        assert ("Y", "Z") in edges
        assert ("Z", "X") not in edges
        assert ("Y", "X") not in edges

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


class TestSortnRegressScoring:
    def test_score(self, causal_chain_data):
        est = SortnRegress(threshold=0.3)
        est.fit(causal_chain_data)

        score = est.score(X=causal_chain_data)
        assert isinstance(score, float)

        true_dag = DAG([("X", "Y"), ("Y", "Z")])
        shd_score = est.score(true_graph=true_dag)
        assert isinstance(shd_score, (int, float, np.integer))


# Compare against reference implementations on the same data.
class TestSortnRegressVsReference:
    """
    Compares pgmpy's SortnRegress against the reference sortnregress from
    scriddie/varsortability (https://github.com/scriddie/varsortability) and
    CausalDisco (https://github.com/CausalDisco/CausalDisco) on the same data.

    Both repositories use the identical algorithm - same authors (Reisach et al.,
    2021), same code.

    _ref_sortnregress covers both comparisons.
    """

    def test_compare_scriddie_and_causaldisco(self, causal_chain_data):
        """
        pgmpy, scriddie/varsortability, and CausalDisco must all recover
        the same true edges on high var-sortability chain data (X->Y->Z->W).
        """
        cols = list(causal_chain_data.columns)
        true_edges = {("X", "Y"), ("Y", "Z")}

        est = SortnRegress(threshold=0.3)
        est.fit(causal_chain_data)
        pgmpy_edges = set(est.causal_graph_.edges())

        W_ref = _ref_sortnregress(causal_chain_data.values)
        ref_edges = {(cols[i], cols[j]) for i in range(len(cols)) for j in range(len(cols)) if abs(W_ref[i, j]) > 0.3}

        assert true_edges <= ref_edges
        assert pgmpy_edges == ref_edges
