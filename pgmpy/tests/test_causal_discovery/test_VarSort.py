import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import VarSort


@pytest.fixture
def causal_chain_data():
    rng = np.random.default_rng(seed=42)
    n = 1000
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, 1.5, n)
    z = y + rng.normal(0, 2.0, n)

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.mark.parametrize("return_type", ["dag", "pdag"])
def test_chain(causal_chain_data, return_type):
    original = causal_chain_data.copy()
    est = VarSort(return_type=return_type)

    assert est.fit(causal_chain_data) is est
    assert est.n_features_in_ == 3
    assert list(est.feature_names_in_) == ["X", "Y", "Z"]
    assert est.causal_order_ == ["X", "Y", "Z"]
    assert set(est.causal_graph_.nodes()) == {"X", "Y", "Z"}
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    if return_type == "dag":
        assert isinstance(est.causal_graph_, DAG)
        assert set(est.causal_graph_.edges()) == {("X", "Y"), ("Y", "Z")}
    else:
        assert isinstance(est.causal_graph_, PDAG)
        assert not est.causal_graph_.directed_edges
        assert est.causal_graph_.undirected_edges == {("X", "Y"), ("Y", "Z")}
        adjacency = adjacency + adjacency.T
    pd.testing.assert_frame_equal(
        est.adjacency_matrix_,
        pd.DataFrame(adjacency, index=causal_chain_data.columns, columns=causal_chain_data.columns),
    )
    pd.testing.assert_frame_equal(causal_chain_data, original)


@pytest.mark.parametrize("return_type", ["dag", "pdag"])
def test_collider(return_type):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["X", "Y", "Z"])
    data["Z"] += 2.5 * data["X"] + 2.5 * data["Y"]
    graph = VarSort(return_type=return_type).fit(data[["Z", "Y", "X"]]).causal_graph_

    if return_type == "dag":
        assert set(graph.edges()) == {("X", "Z"), ("Y", "Z")}
    else:
        assert graph.directed_edges == {("X", "Z"), ("Y", "Z")}
        assert not graph.undirected_edges


def test_scale_sensitivity(causal_chain_data):
    raw = VarSort().fit(causal_chain_data)
    scaled = causal_chain_data * [100, 1, 0.01]
    rescaled = VarSort().fit(scaled)

    assert raw.causal_order_ != rescaled.causal_order_
    assert set(raw.causal_graph_.edges()) != set(rescaled.causal_graph_.edges())
