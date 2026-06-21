"""
Tests for DiBS in pgmpy.causal_discovery.
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.utils.estimator_checks import parametrize_with_checks

torch = _safe_import("torch")

if _check_soft_dependencies("torch", severity="none"):
    from pgmpy.causal_discovery.DiBS import DiBS

def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [DiBS(n_steps=2, n_particles=2, n_grad_mc_samples=2, n_acyclicity_mc_samples=2)],
    expected_failed_checks=expected_failed_checks,
)
def test_DiBS_compatibility(estimator, check):
    check(estimator)


def linear_chain_data():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(scale=0.1, size=n)
    c = -1.5 * b + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def tiny_data():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    return pd.DataFrame(x, columns=["X1", "X2", "X3"])


@unittest.skipUnless(
    _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestDiBSCore(unittest.TestCase):
    def test_fit_sets_attributes(self):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
            edge_prob_threshold=0.0,
        )
        est.fit(linear_chain_data())

        self.assertTrue(hasattr(est, "causal_graph_"))
        self.assertTrue(hasattr(est, "edge_probs_"))
        self.assertTrue(hasattr(est, "adjacency_matrix_"))
        self.assertTrue(hasattr(est, "n_features_in_"))
        self.assertTrue(hasattr(est, "feature_names_in_"))

        self.assertIsInstance(est.causal_graph_, nx.DiGraph)
        self.assertIsInstance(est.edge_probs_, pd.DataFrame)
        self.assertIsInstance(est.adjacency_matrix_, pd.DataFrame)

        self.assertEqual(est.n_features_in_, 3)
        self.assertEqual(est.feature_names_in_.tolist(), ["A", "B", "C"])
        self.assertEqual(est.edge_probs_.shape, (3, 3))
        self.assertEqual(est.adjacency_matrix_.shape, (3, 3))

    def test_edge_probs_diagonal_zero(self):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
        )
        est.fit(linear_chain_data())
        self.assertTrue(np.allclose(np.diag(est.edge_probs_.to_numpy()), 0.0))

    def test_adjacency_matrix_binary(self):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
            edge_prob_threshold=0.0,
        )
        est.fit(linear_chain_data())
        vals = np.unique(est.adjacency_matrix_.to_numpy())
        self.assertTrue(set(vals).issubset({0, 1}))

    def test_summary_graph_acyclic(self):
        est = DiBS(
            n_particles=6,
            n_steps=2,
            latent_dim=8,
            n_grad_mc_samples=8,
            n_acyclicity_mc_samples=4,
            edge_prob_threshold=0.0,
        )
        est.fit(linear_chain_data())
        self.assertTrue(nx.is_directed_acyclic_graph(est.causal_graph_))

    def test_custom_log_likelihood_callable(self):
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
        est.fit(tiny_data())
        self.assertGreater(calls["count"], 0)


@unittest.skipUnless(
    _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestDiBSLikelihoodValidation(unittest.TestCase):
    def test_unknown_grad_estimator_raises(self):
        est = DiBS(grad_estimator_z="Nonsense estimator")
        with self.assertRaisesRegex(ValueError, "Unknown grad estimator"):
            est.fit(linear_chain_data())
