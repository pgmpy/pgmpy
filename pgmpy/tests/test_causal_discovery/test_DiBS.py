
"""
Tests for DiBS in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from pgmpy.causal_discovery.DiBS import DiBS


@pytest.fixture
def linear_chain_data():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(scale=0.1, size=n)
    c = -1.5 * b + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    return pd.DataFrame(x, columns=["X1", "X2", "X3"])


class TestDiBSCore:
    def test_fit_sets_attributes(self, linear_chain_data):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
            edge_prob_threshold=0.0,
        )
        est.fit(linear_chain_data)

        assert hasattr(est, "causal_graph_")
        assert hasattr(est, "edge_probs_")
        assert hasattr(est, "adjacency_matrix_")
        assert hasattr(est, "n_features_in_")
        assert hasattr(est, "feature_names_in_")

        assert isinstance(est.causal_graph_, nx.DiGraph)
        assert isinstance(est.edge_probs_, pd.DataFrame)
        assert isinstance(est.adjacency_matrix_, pd.DataFrame)

        assert est.n_features_in_ == 3
        assert est.feature_names_in_ == ["A", "B", "C"]
        assert est.edge_probs_.shape == (3, 3)
        assert est.adjacency_matrix_.shape == (3, 3)

    def test_edge_probs_diagonal_zero(self, linear_chain_data):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
        )
        est.fit(linear_chain_data)
        assert np.allclose(np.diag(est.edge_probs_.to_numpy()), 0.0)

    def test_adjacency_matrix_binary(self, linear_chain_data):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
            edge_prob_threshold=0.0,
        )
        est.fit(linear_chain_data)
        vals = np.unique(est.adjacency_matrix_.to_numpy())
        assert set(vals).issubset({0, 1})

    def test_summary_graph_acyclic(self, linear_chain_data):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
            edge_prob_threshold=0.0,
        )
        est.fit(linear_chain_data)
        assert nx.is_directed_acyclic_graph(est.causal_graph_)

    def test_sample_graphs_raises_before_inference(self):
        est = DiBS(n_particles=3, n_steps=1)
        with pytest.raises(ValueError, match="No graph particle samples found"):
            est._sample_graphs(["A", "B"])

    def test_custom_log_likelihood_callable(self, tiny_data):
        calls = {"count": 0}

        def custom_ll(data: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
            calls["count"] += 1
            out_shape = graph.shape[:-2]
            if len(out_shape) == 0:
                return torch.zeros((), dtype=data.dtype, device=graph.device)
            return torch.zeros(out_shape, dtype=data.dtype, device=graph.device)

        est = DiBS(
            n_particles=4,
            n_steps=1,
            latent_dim=4,
            n_grad_mc_samples=4,
            n_acyclicity_mc_samples=2,
            log_likelihood=custom_ll,
        )
        est.fit(tiny_data)
        assert calls["count"] > 0


class TestDiBSSummarizeGraphs:
    def test_summarize_graphs_empty_raises(self):
        est = DiBS()
        with pytest.raises(ValueError, match="must contain at least one graph"):
            est._summarize_graphs([])

    def test_summarize_graphs_tensor_input(self):
        est = DiBS(edge_prob_threshold=0.5)

        # 2 posterior samples over 3 nodes
        samples = torch.tensor(
            [
                [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        )

        summary_graph, edge_probs, adjacency_matrix = est._summarize_graphs(samples)

        assert isinstance(summary_graph, nx.DiGraph)
        assert isinstance(edge_probs, pd.DataFrame)
        assert isinstance(adjacency_matrix, pd.DataFrame)

        assert edge_probs.shape == (3, 3)
        assert adjacency_matrix.shape == (3, 3)

        # Edge 0->1 appears in both samples
        assert edge_probs.iloc[0, 1] == pytest.approx(1.0)
        assert adjacency_matrix.iloc[0, 1] == 1

    def test_summarize_graphs_inconsistent_node_order_raises(self):
        est = DiBS()

        g1 = nx.DiGraph()
        g1.add_nodes_from(["A", "B", "C"])

        g2 = nx.DiGraph()
        g2.add_nodes_from(["B", "A", "C"])  # different insertion order

        with pytest.raises(ValueError, match="same node ordering"):
            est._summarize_graphs([g1, g2])


class TestDiBSLikelihoodValidation:
    def test_lgbn_log_likelihood_shape_checks(self):
        est = DiBS()
        data = torch.randn(20, 3)

        with pytest.raises(ValueError, match="`data` must have shape"):
            est._lgbn_log_likelihood(torch.randn(20, 3, 1), torch.zeros(3, 3))

        with pytest.raises(ValueError, match="`graph` must have shape"):
            est._lgbn_log_likelihood(data, torch.zeros(3))

        with pytest.raises(ValueError, match="Mismatch: data has 3 variables"):
            est._lgbn_log_likelihood(data, torch.zeros(4, 4))

    def test_unknown_grad_estimator_raises(self):
        est = DiBS()
        with pytest.raises(ValueError, match="Unknown grad estimator"):
            est._make_likelihood_grad_estimator("invalid")