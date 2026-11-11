import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pgmpy.base import DAG
from pgmpy.causal_discovery import DirectLiNGAM


@pytest.fixture
def synthetic_3node_data():
    rng = np.random.default_rng(42)
    n = 2000
    x0 = rng.uniform(-2, 2, n)
    x1 = 1.5 * x0 + rng.laplace(0, 1, n)
    x2 = 0.8 * x1 + rng.uniform(-1, 1, n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


@pytest.fixture
def synthetic_4node_data():
    rng = np.random.default_rng(42)
    n = 3000
    x0 = rng.uniform(-2, 2, n)
    x1 = 1.2 * x0 + rng.laplace(0, 1, n)
    x2 = 1.5 * x0 + 0.8 * x1 + rng.uniform(-1, 1, n)
    x3 = 0.9 * x2 + rng.laplace(0, 0.5, n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3})


def test_3node_linear_dag(synthetic_3node_data):
    """Test DirectLiNGAM on a 3-node chain DAG with uniform and Laplace noise."""
    est = DirectLiNGAM(random_state=42, threshold=0.1)
    res = est.fit(synthetic_3node_data)

    assert res is est
    assert est.n_features_in_ == 3
    assert list(est.feature_names_in_) == ["X0", "X1", "X2"]
    assert est.causal_order_ == ["X0", "X1", "X2"]

    # Verify causal_graph_ is a DAG without false cycles
    assert isinstance(est.causal_graph_, DAG)
    assert nx.is_directed_acyclic_graph(est.causal_graph_)
    assert set(est.causal_graph_.edges()) == {("X0", "X1"), ("X1", "X2")}

    # Verify adjacency_matrix_ matches expected topological order
    assert isinstance(est.adjacency_matrix_, pd.DataFrame)
    assert est.adjacency_matrix_.loc["X0", "X1"] == 1
    assert est.adjacency_matrix_.loc["X1", "X2"] == 1
    assert est.adjacency_matrix_.loc["X1", "X0"] == 0
    assert est.adjacency_matrix_.loc["X2", "X1"] == 0
    assert est.adjacency_matrix_.loc["X0", "X2"] == 0
    assert est.adjacency_matrix_.loc["X2", "X0"] == 0

    # In topological order, all entries on and below diagonal must be zero
    order = est.causal_order_
    adj_ordered = est.adjacency_matrix_.loc[order, order].values
    assert np.all(np.tril(adj_ordered) == 0)

    # Verify B_ matrix
    assert isinstance(est.B_, pd.DataFrame)
    b_ordered = est.B_.loc[order, order].values
    assert np.all(np.triu(b_ordered) == 0)
    assert pytest.approx(est.B_.loc["X1", "X0"], rel=0.1) == 1.5
    assert pytest.approx(est.B_.loc["X2", "X1"], rel=0.1) == 0.8


def test_3node_shuffled_columns(synthetic_3node_data):
    """Test that input column order does not affect causal order discovery."""
    shuffled_data = synthetic_3node_data[["X2", "X0", "X1"]]
    est = DirectLiNGAM(random_state=42, threshold=0.1).fit(shuffled_data)

    assert est.causal_order_ == ["X0", "X1", "X2"]
    assert isinstance(est.causal_graph_, DAG)
    assert nx.is_directed_acyclic_graph(est.causal_graph_)
    assert set(est.causal_graph_.edges()) == {("X0", "X1"), ("X1", "X2")}

    order = est.causal_order_
    adj_ordered = est.adjacency_matrix_.loc[order, order].values
    assert np.all(np.tril(adj_ordered) == 0)


def test_4node_linear_dag(synthetic_4node_data):
    """Test DirectLiNGAM on a 4-node DAG with multiple paths."""
    est = DirectLiNGAM(random_state=42, threshold=0.1).fit(synthetic_4node_data)

    assert est.n_features_in_ == 4
    assert est.causal_order_ == ["X0", "X1", "X2", "X3"]
    assert isinstance(est.causal_graph_, DAG)
    assert nx.is_directed_acyclic_graph(est.causal_graph_)

    expected_edges = {("X0", "X1"), ("X0", "X2"), ("X1", "X2"), ("X2", "X3")}
    assert set(est.causal_graph_.edges()) == expected_edges

    order = est.causal_order_
    adj_ordered = est.adjacency_matrix_.loc[order, order].values
    assert np.all(np.tril(adj_ordered) == 0)

    b_ordered = est.B_.loc[order, order].values
    assert np.all(np.triu(b_ordered) == 0)


@pytest.mark.parametrize("measure", ["kernel", "mutual_info"])
def test_alternative_measures(synthetic_3node_data, measure):
    """Test alternative independence measures: kernel and mutual_info."""
    est = DirectLiNGAM(random_state=42, measure=measure, threshold=0.1).fit(synthetic_3node_data)

    assert est.causal_order_ == ["X0", "X1", "X2"]
    assert isinstance(est.causal_graph_, DAG)
    assert nx.is_directed_acyclic_graph(est.causal_graph_)
    assert set(est.causal_graph_.edges()) == {("X0", "X1"), ("X1", "X2")}


def test_threshold_pruning():
    """Test that weak coefficients are pruned according to threshold."""
    rng = np.random.default_rng(42)
    n = 2000
    x0 = rng.uniform(-2, 2, n)
    x1 = 0.04 * x0 + rng.laplace(0, 1, n)  # weak edge
    data = pd.DataFrame({"X0": x0, "X1": x1})

    # With threshold 0.1, edge should be pruned
    est_pruned = DirectLiNGAM(random_state=42, threshold=0.1).fit(data)
    assert len(est_pruned.causal_graph_.edges()) == 0
    assert est_pruned.adjacency_matrix_.loc["X0", "X1"] == 0

    # With threshold 0.01, edge should be kept
    est_kept = DirectLiNGAM(random_state=42, threshold=0.01).fit(data)
    assert ("X0", "X1") in est_kept.causal_graph_.edges()
    assert est_kept.adjacency_matrix_.loc["X0", "X1"] == 1


def test_sklearn_compatibility(synthetic_3node_data):
    """Test Scikit-learn compatibility: clone and fit on continuous DataFrame."""
    est = DirectLiNGAM(random_state=42, measure="pwling", threshold=0.15)
    cloned_est = clone(est)

    assert cloned_est.random_state == 42
    assert cloned_est.measure == "pwling"
    assert cloned_est.threshold == 0.15
    assert not hasattr(cloned_est, "causal_order_")

    cloned_est.fit(synthetic_3node_data)
    assert hasattr(cloned_est, "causal_order_")
    assert hasattr(cloned_est, "causal_graph_")
    assert hasattr(cloned_est, "adjacency_matrix_")
    assert hasattr(cloned_est, "B_")

    # Tags check
    tags = cloned_est.__sklearn_tags__()
    assert tags.input_tags.categorical is False


def test_validation_and_edge_cases():
    """Test input validations and edge cases."""
    rng = np.random.default_rng(42)

    # Invalid measure
    with pytest.raises(ValueError, match="measure must be one of"):
        DirectLiNGAM(measure="invalid").fit(pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}))

    # Negative threshold
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        DirectLiNGAM(threshold=-0.1).fit(pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}))

    # Discrete / categorical data
    discrete_data = pd.DataFrame({"A": list("aabbab"), "B": list("xyxyxy")})
    with pytest.raises(ValueError, match="continuous"):
        DirectLiNGAM().fit(discrete_data)

    # Single feature
    single_col_data = pd.DataFrame({"X": rng.uniform(-1, 1, 50)})
    est_single = DirectLiNGAM().fit(single_col_data)
    assert est_single.causal_order_ == ["X"]
    assert len(est_single.causal_graph_.nodes()) == 1
    assert len(est_single.causal_graph_.edges()) == 0
