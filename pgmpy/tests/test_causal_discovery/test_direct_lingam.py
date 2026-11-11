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
    with pytest.raises(ValueError, match="threshold must be non-negative and finite"):
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


def test_integer_column_names(synthetic_3node_data):
    """Test that DirectLiNGAM fits cleanly and returns valid outputs with integer column names."""
    data = synthetic_3node_data.copy()
    data.columns = [0, 1, 2]
    est = DirectLiNGAM(random_state=42, threshold=0.1).fit(data)

    assert est.causal_order_ == [0, 1, 2]
    assert isinstance(est.causal_graph_, DAG)
    assert set(est.causal_graph_.nodes()) == {0, 1, 2}
    assert set(est.causal_graph_.edges()) == {(0, 1), (1, 2)}

    # Verify B_ matrix with integer indices and columns
    assert isinstance(est.B_, pd.DataFrame)
    assert list(est.B_.index) == [0, 1, 2]
    assert list(est.B_.columns) == [0, 1, 2]
    assert pytest.approx(est.B_.loc[1, 0], rel=0.1) == 1.5
    assert pytest.approx(est.B_.loc[2, 1], rel=0.1) == 0.8

    # Verify adjacency_matrix_ with integer indices and columns
    assert isinstance(est.adjacency_matrix_, pd.DataFrame)
    assert list(est.adjacency_matrix_.index) == [0, 1, 2]
    assert list(est.adjacency_matrix_.columns) == [0, 1, 2]
    assert est.adjacency_matrix_.loc[0, 1] == 1
    assert est.adjacency_matrix_.loc[1, 2] == 1
    assert est.adjacency_matrix_.loc[1, 0] == 0
    assert est.adjacency_matrix_.loc[2, 0] == 0

    # Single integer column
    single_int = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0]})
    est_single = DirectLiNGAM().fit(single_int)
    assert est_single.causal_order_ == [0]
    assert list(est_single.causal_graph_.nodes()) == [0]
    assert est_single.B_.shape == (1, 1)
    assert list(est_single.B_.columns) == [0]


def test_uniformly_small_scale():
    """Test DirectLiNGAM on dataset scaled by 1e-7 where X -> Y -> Z."""
    rng = np.random.default_rng(42)
    n = 2000
    scale = 1e-7
    x = rng.uniform(-2, 2, n) * scale
    y = 1.5 * x + rng.laplace(0, 1, n) * scale
    z = 0.8 * y + rng.uniform(-1, 1, n) * scale
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})

    est = DirectLiNGAM(random_state=42, threshold=0.1).fit(data)

    assert est.causal_order_ == ["X", "Y", "Z"]
    assert isinstance(est.causal_graph_, DAG)
    assert set(est.causal_graph_.edges()) == {("X", "Y"), ("Y", "Z")}
    assert pytest.approx(est.B_.loc["Y", "X"], rel=0.1) == 1.5
    assert pytest.approx(est.B_.loc["Z", "Y"], rel=0.1) == 0.8
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "Z"] == 1
    assert est.adjacency_matrix_.loc["X", "Z"] == 0


def test_edge_creation_zero_threshold():
    """Test that true zero coefficients or unrelated noise do not produce edges with threshold=0.0."""
    rng = np.random.default_rng(42)
    n = 2000
    x0 = rng.uniform(-2, 2, n)
    x1 = 1.5 * x0 + rng.laplace(0, 1, n)

    # Gram-Schmidt orthogonalization to ensure partial coefficient on x0 is identically zero
    def _project_out(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        return v - (np.dot(v, u) / np.dot(u, u)) * u

    u0 = np.ones(n)
    u1 = _project_out(x0, u0)
    u2 = _project_out(_project_out(x1, u0), u1)

    raw_noise = rng.uniform(-1, 1, n)
    ortho_noise = _project_out(_project_out(_project_out(raw_noise, u0), u1), u2)
    x2 = 0.8 * x1 + ortho_noise

    data = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
    est = DirectLiNGAM(random_state=42, threshold=0.0).fit(data)

    assert est.causal_order_ == ["X0", "X1", "X2"]
    # (X0, X2) edge must not be created because true partial coefficient is 0.0
    assert ("X0", "X2") not in est.causal_graph_.edges()
    assert set(est.causal_graph_.edges()) == {("X0", "X1"), ("X1", "X2")}
    assert est.B_.loc["X2", "X0"] == 0.0
    assert est.adjacency_matrix_.loc["X0", "X2"] == 0

    # Independent orthogonal noise variable: no edges should be created to/from it
    raw_indep = rng.uniform(-1, 1, n)
    x1_indep = _project_out(_project_out(raw_indep, u0), u1)
    df_indep = pd.DataFrame({"X0": x0, "X1": x1_indep})
    est_indep = DirectLiNGAM(random_state=42, threshold=0.0).fit(df_indep)

    assert len(est_indep.causal_graph_.edges()) == 0
    assert est_indep.B_.loc["X1", "X0"] == 0.0
    assert est_indep.B_.loc["X0", "X1"] == 0.0
    assert np.all(est_indep.adjacency_matrix_.values == 0)


@pytest.mark.parametrize("invalid_threshold", [None, np.nan, np.inf, -0.1])
def test_threshold_validation(invalid_threshold):
    """Test that threshold=None, np.nan, np.inf, and -0.1 raise ValueError."""
    data = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    with pytest.raises(ValueError, match="threshold must be non-negative and finite"):
        DirectLiNGAM(threshold=invalid_threshold).fit(data)


@pytest.mark.parametrize(
    "rs",
    [
        np.random.RandomState(42),
        np.random.default_rng(42),
        42,
        None,
    ],
)
def test_random_state_support(synthetic_3node_data, rs):
    """Test fitting with RandomState, Generator, int, and None."""
    est = DirectLiNGAM(random_state=rs, threshold=0.1).fit(synthetic_3node_data)
    assert est.causal_order_ == ["X0", "X1", "X2"]
    assert set(est.causal_graph_.edges()) == {("X0", "X1"), ("X1", "X2")}


@pytest.mark.parametrize(
    "rs_factory",
    [
        lambda: np.random.RandomState(42),
        lambda: np.random.default_rng(42),
    ],
)
def test_mutual_info_reproducibility(synthetic_3node_data, rs_factory):
    """Verify reproducibility when measure='mutual_info' with RandomState or Generator."""
    est1 = DirectLiNGAM(random_state=rs_factory(), measure="mutual_info", threshold=0.1).fit(synthetic_3node_data)
    est2 = DirectLiNGAM(random_state=rs_factory(), measure="mutual_info", threshold=0.1).fit(synthetic_3node_data)

    assert est1.causal_order_ == est2.causal_order_
    assert set(est1.causal_graph_.edges()) == set(est2.causal_graph_.edges())
    pd.testing.assert_frame_equal(est1.B_, est2.B_)
    pd.testing.assert_frame_equal(est1.adjacency_matrix_, est2.adjacency_matrix_)
