"""
Tests for the sklearn-compatible MMHC class in pgmpy.causal_discovery.

MMHC is discrete-only (uses chi-square CI tests). Sklearn's full estimator_checks
use arbitrary dtypes (e.g. int32 treated as continuous), so we run a subset of
compatibility tests with discrete data instead of parametrize_with_checks.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pgmpy.causal_discovery import MMHC


def test_estimator_clone():
    """Estimator can be cloned (sklearn contract)."""
    est = MMHC(scoring_method="bdeu")
    cloned = clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()


def test_estimator_get_set_params():
    """get_params and set_params work (sklearn contract)."""
    est = MMHC(significance_level=0.05, tabu_length=5)
    params = est.get_params()
    assert params["significance_level"] == 0.05
    assert params["tabu_length"] == 5
    est.set_params(significance_level=0.001)
    assert est.significance_level == 0.001


def test_discrete_data_required(xyz_sum_data):
    """MMHC requires discrete data; continuous/mixed raises clear error."""
    # Integer dtype without category is often treated as continuous by pgmpy
    X_cont = pd.DataFrame(
        np.random.randn(100, 3), columns=list("ABC")
    )  # float = continuous
    with pytest.raises(ValueError, match="MMHC only supports discrete"):
        MMHC(scoring_method="bdeu", show_progress=False).fit(X_cont)
    # Discrete data (category) works
    MMHC(scoring_method="bdeu", show_progress=False).fit(xyz_sum_data)


@pytest.fixture
def xyz_sum_data():
    """Data with X, Y, Z and sum = X + Y + Z (discrete)."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randint(0, 2, size=(int(1e4), 3)), columns=list("XYZ")
    )
    data["sum"] = data.sum(axis=1)
    return data.astype("category")


@pytest.fixture
def xyz_sum_data_large():
    """Larger sample for more stable skeleton (like original MmhcEstimator test)."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randint(0, 2, size=(int(1e5), 3)), columns=list("XYZ")
    )
    data["sum"] = data.sum(axis=1)
    return data.astype("category")


def test_fit_returns_self(xyz_sum_data):
    mmhc = MMHC(scoring_method="bdeu", significance_level=0.01, show_progress=False)
    out = mmhc.fit(xyz_sum_data)
    assert out is mmhc


def test_fit_sets_attributes(xyz_sum_data):
    mmhc = MMHC(scoring_method="bdeu", significance_level=0.01, show_progress=False)
    mmhc.fit(xyz_sum_data)
    assert hasattr(mmhc, "causal_graph_")
    assert hasattr(mmhc, "skeleton_")
    assert hasattr(mmhc, "adjacency_matrix_")
    assert hasattr(mmhc, "n_features_in_")
    assert hasattr(mmhc, "feature_names_in_")
    assert mmhc.n_features_in_ == 4
    assert list(mmhc.feature_names_in_) == ["X", "Y", "Z", "sum"]


def test_estimate_edges_subset_of_possible(xyz_sum_data_large):
    """Learned DAG edges should be a subset of possible skeleton orientations."""
    mmhc = MMHC(
        scoring_method="bdeu",
        significance_level=0.01,
        tabu_length=10,
        return_type="dag",
        show_progress=False,
    )
    mmhc.fit(xyz_sum_data_large)
    possible_edges = {
        ("X", "sum"),
        ("Y", "sum"),
        ("Z", "sum"),
        ("sum", "X"),
        ("sum", "Y"),
        ("sum", "Z"),
        ("X", "Y"),
        ("X", "Z"),
        ("Y", "Z"),
        ("Y", "X"),
        ("Z", "X"),
        ("Z", "Y"),
    }
    assert set(mmhc.causal_graph_.edges()).issubset(possible_edges)
    assert len(mmhc.causal_graph_.edges()) >= 1


def test_skeleton_is_undirected(xyz_sum_data):
    mmhc = MMHC(scoring_method="bdeu", significance_level=0.01, show_progress=False)
    mmhc.fit(xyz_sum_data)
    assert set(mmhc.skeleton_.nodes()) == set(xyz_sum_data.columns)
    assert mmhc.skeleton_.number_of_nodes() == xyz_sum_data.shape[1]
    # Causal graph edges (as unordered pairs) should be subset of skeleton edges
    skeleton_pairs = {frozenset([u, v]) for u, v in mmhc.skeleton_.edges()}
    for u, v in mmhc.causal_graph_.edges():
        assert frozenset([u, v]) in skeleton_pairs


def test_return_type_dag(xyz_sum_data):
    mmhc = MMHC(
        scoring_method="bdeu",
        return_type="dag",
        show_progress=False,
    )
    mmhc.fit(xyz_sum_data)
    assert mmhc.causal_graph_.__class__.__name__ == "DAG"


def test_return_type_pdag(xyz_sum_data):
    mmhc = MMHC(
        scoring_method="bdeu",
        return_type="pdag",
        show_progress=False,
    )
    mmhc.fit(xyz_sum_data)
    assert mmhc.causal_graph_.__class__.__name__ == "PDAG"


def test_significance_level_sparser(xyz_sum_data):
    """Stricter significance level should yield sparser or same skeleton."""
    mmhc_01 = MMHC(
        scoring_method="bdeu",
        significance_level=0.01,
        show_progress=False,
    )
    mmhc_001 = MMHC(
        scoring_method="bdeu",
        significance_level=0.001,
        show_progress=False,
    )
    mmhc_01.fit(xyz_sum_data)
    mmhc_001.fit(xyz_sum_data)
    # 0.001 is stricter: fewer edges accepted, so skeleton should have <= edges
    assert len(mmhc_001.skeleton_.edges()) <= len(mmhc_01.skeleton_.edges())


@pytest.mark.parametrize("scoring_method", ["k2", "bdeu", "bic-d"])
def test_scoring_methods_discrete(xyz_sum_data, scoring_method):
    mmhc = MMHC(
        scoring_method=scoring_method,
        return_type="dag",
        show_progress=False,
    )
    mmhc.fit(xyz_sum_data)
    assert mmhc.causal_graph_.number_of_nodes() == 4
