"""
Tests for the sklearn-compatible MMHC class in pgmpy.causal_discovery.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from pgmpy.causal_discovery import MMHC
from pgmpy.estimators import MmhcEstimator
from pgmpy.metrics import CorrelationScore
from pgmpy.utils import get_example_model


# ---------------------------------------------------------------------------
# sklearn compatibility checks
# ---------------------------------------------------------------------------


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": (
            "Causal discovery estimators do not take y parameter in score method."
        ),
        "check_n_features_in_after_fitting": (
            "Failing for score method (not for fit) for unknown reason."
        ),
    }


@parametrize_with_checks(
    [MMHC(return_type="dag", show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
def test_mmhc_compatibility(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_data():
    """Small discrete dataset where X, Y, Z are causes of 'sum'."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.integers(0, 2, size=(2000, 3)), columns=list("XYZ")
    )
    df["sum"] = df.sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Fit behaviour
# ---------------------------------------------------------------------------


def test_fit_sets_attributes(simple_data):
    """fit() must populate causal_graph_, skeleton_, and adjacency_matrix_."""
    est = MMHC(show_progress=False)
    est.fit(simple_data)

    assert hasattr(est, "causal_graph_"), "causal_graph_ not set after fit"
    assert hasattr(est, "skeleton_"), "skeleton_ not set after fit"
    assert hasattr(est, "adjacency_matrix_"), "adjacency_matrix_ not set after fit"
    assert hasattr(est, "n_features_in_"), "n_features_in_ not set after fit"


def test_fit_returns_self(simple_data):
    """fit() must return self (sklearn contract)."""
    est = MMHC(show_progress=False)
    result = est.fit(simple_data)
    assert result is est


def test_causal_graph_has_correct_nodes(simple_data):
    """causal_graph_ must contain all data columns as nodes."""
    est = MMHC(show_progress=False)
    est.fit(simple_data)
    assert set(est.causal_graph_.nodes()) == set(simple_data.columns)


def test_skeleton_is_undirected(simple_data):
    """skeleton_ must be an UndirectedGraph whose nodes match the data columns."""
    from pgmpy.base import UndirectedGraph

    est = MMHC(show_progress=False)
    est.fit(simple_data)
    assert isinstance(est.skeleton_, UndirectedGraph)
    assert set(est.skeleton_.nodes()) == set(simple_data.columns)


def test_adjacency_matrix_shape(simple_data):
    """adjacency_matrix_ must be a square DataFrame with data columns as index/columns."""
    est = MMHC(show_progress=False)
    est.fit(simple_data)
    adj = est.adjacency_matrix_
    assert isinstance(adj, pd.DataFrame)
    assert adj.shape == (len(simple_data.columns), len(simple_data.columns))
    assert list(adj.columns) == list(adj.index)


# ---------------------------------------------------------------------------
# sklearn clone compatibility
# ---------------------------------------------------------------------------


def test_sklearn_clone():
    """clone(MMHC()) must work — constructor stores ONLY hyperparameters."""
    cloned = clone(MMHC())
    assert cloned is not None


def test_sklearn_clone_does_not_share_state(simple_data):
    """Cloned estimator must be independent (no shared fitted state)."""
    est = MMHC(show_progress=False)
    est.fit(simple_data)

    cloned = clone(est)
    # The clone should not be fitted
    with pytest.raises(NotFittedError):
        check_is_fitted(cloned, "causal_graph_")


# ---------------------------------------------------------------------------
# NotFittedError
# ---------------------------------------------------------------------------


def test_score_not_fitted_raises(simple_data):
    """score() before fit() must raise NotFittedError."""
    est = MMHC(show_progress=False)
    with pytest.raises(NotFittedError):
        est.score(X=simple_data)


# ---------------------------------------------------------------------------
# score() method
# ---------------------------------------------------------------------------


def test_score_after_fit(simple_data):
    """score() should return a numeric value after fit()."""
    est = MMHC(return_type="dag", show_progress=False)
    est.fit(simple_data)
    score = est.score(X=simple_data)
    assert isinstance(score, float)


def test_score_with_true_graph():
    """score(true_graph=...) should work for graph-comparison metrics."""
    asia_model = get_example_model("asia")
    data = asia_model.simulate(n_samples=int(1e3), seed=42)

    est = MMHC(return_type="dag", show_progress=False)
    est.fit(data)

    shd = est.score(true_graph=asia_model)
    assert shd is not None


def test_score_with_metric_instance(simple_data):
    """score() should accept a metric instance."""
    est = MMHC(return_type="dag", show_progress=False)
    est.fit(simple_data)
    corr = est.score(X=simple_data, metric=CorrelationScore(significance_level=0.01))
    assert isinstance(corr, float)


def test_score_with_metric_string(simple_data):
    """score() should accept a metric identified by string name."""
    est = MMHC(return_type="dag", show_progress=False)
    est.fit(simple_data)
    score = est.score(X=simple_data, metric="structure_score")
    assert isinstance(score, float)


def test_score_raises_without_x_or_graph(simple_data):
    """score() without any argument must raise ValueError."""
    est = MMHC(show_progress=False)
    est.fit(simple_data)
    with pytest.raises(ValueError):
        est.score()


# ---------------------------------------------------------------------------
# return_type parameter
# ---------------------------------------------------------------------------


def test_return_type_dag(simple_data):
    """return_type='dag' must produce a DAG instance."""
    from pgmpy.base import DAG

    est = MMHC(return_type="dag", show_progress=False)
    est.fit(simple_data)
    assert isinstance(est.causal_graph_, DAG)


def test_return_type_pdag(simple_data):
    """return_type='pdag' must produce a PDAG instance."""
    from pgmpy.base import PDAG

    est = MMHC(return_type="pdag", show_progress=False)
    est.fit(simple_data)
    assert isinstance(est.causal_graph_, PDAG)


# ---------------------------------------------------------------------------
# Significance level effect on skeleton sparsity
# ---------------------------------------------------------------------------


def test_significance_level_affects_skeleton(simple_data):
    """Lower significance_level should produce a sparser or equal skeleton."""
    est_default = MMHC(significance_level=0.05, show_progress=False)
    est_strict = MMHC(significance_level=0.001, show_progress=False)

    est_default.fit(simple_data)
    est_strict.fit(simple_data)

    # Stricter threshold should be at most as dense as the looser one
    assert est_strict.skeleton_.number_of_edges() <= est_default.skeleton_.number_of_edges()


# ---------------------------------------------------------------------------
# Deprecation warning on old MmhcEstimator
# ---------------------------------------------------------------------------


def test_old_mmhcestimator_raises_deprecation_warning(simple_data):
    """Using the old MmhcEstimator constructor must emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MmhcEstimator(simple_data)

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) > 0, "Expected a DeprecationWarning"
    assert "pgmpy.causal_discovery.MMHC" in str(deprecation_warnings[0].message)


def test_old_mmhcestimator_still_functional(simple_data):
    """Despite deprecation, the old MmhcEstimator.estimate() should still work."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        est = MmhcEstimator(simple_data)
        model = est.estimate()

    assert model is not None
    assert set(model.nodes()) == set(simple_data.columns)
