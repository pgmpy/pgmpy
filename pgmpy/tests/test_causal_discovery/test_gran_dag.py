"""
Tests for the GraNDAG class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery import GraNDAG
from pgmpy.causal_discovery.gran_dag import (
    GraNDAGNetworkConfig,
    GraNDAGRegularizationConfig,
    GraNDAGTrainingConfig,
    _dag_constraint,
)

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
        "check_fit2d_1feature": "GraNDAG requires at least two variables.",
    }


if _check_soft_dependencies("torch", severity="none"):

    @parametrize_with_checks(
        [GraNDAG(max_epochs=1, max_subproblems=1, seed=0)],
        expected_failed_checks=expected_failed_checks,
    )
    def test_gran_dag_compatibility(estimator, check):
        check(estimator)


@pytest.fixture(scope="module")
def numeric_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((12, 3)), columns=["A", "B", "C"])


@pytest.fixture(scope="module")
def fitted_model(numeric_df):
    model = GraNDAG(max_epochs=1, max_subproblems=1, val_size=0, seed=0)
    return model, model.fit(numeric_df)


@requires_torch
def test_dag_constraint_behavior():
    import torch

    # Zero matrix (DAG) should return a scalar tensor with value ~0.0
    U_zero = torch.zeros(4, 4)
    res_zero = _dag_constraint(U_zero)
    assert res_zero.shape == torch.Size([])
    assert res_zero.item() == pytest.approx(0.0, abs=1e-5)

    # Cyclic graph (node 0 -> node 1 -> node 0) should be positive
    U_cyclic = torch.zeros(4, 4)
    U_cyclic[0, 1] = 1.0
    U_cyclic[1, 0] = 1.0
    assert _dag_constraint(U_cyclic).item() > 0.0


class TestOptimizerValidation:
    @requires_torch
    def test_invalid_optimizer_string_raises(self, numeric_df):
        from pgmpy.causal_discovery.gran_dag import GraNDAG

        with pytest.raises(ValueError, match="Supported optimizers are"):
            GraNDAG(optimizer="invalid_optimizer", max_epochs=1).fit(numeric_df)

    @requires_torch
    @pytest.mark.parametrize(
        ("optimizer", "bad_kwargs", "match"),
        [
            ("adam", {"momentum": 0.9}, "Unknown optimizer_params"),
            ("sgd", {"betas": (0.9, 0.999)}, "Unknown optimizer_params"),
            ("adamw", {"nesterov": True}, "Unknown optimizer_params"),
            ("rmsprop", {"nesterov": True}, "Unknown optimizer_params"),
            ("adam", {"params": [1, 2, 3]}, "params"),
        ],
    )
    def test_invalid_kwargs_raises(self, numeric_df, optimizer, bad_kwargs, match):
        from pgmpy.causal_discovery.gran_dag import GraNDAG

        with pytest.raises(ValueError, match=match):
            GraNDAG(optimizer=optimizer, optimizer_params=bad_kwargs, max_epochs=1).fit(numeric_df)


@requires_torch
class TestGraNDAGFit:
    def test_fit_returns_estimator_and_populates_outputs(self, fitted_model, numeric_df):
        model, result = fitted_model
        assert result is model
        assert model.n_features_in_ == numeric_df.shape[1]
        assert list(model.feature_names_in_) == list(numeric_df.columns)
        assert model.model_ is not None
        assert model.scaler_ is not None
        assert isinstance(model.adjacency_matrix_, pd.DataFrame)
        assert isinstance(model.causal_graph_, DAG)
        assert list(model.adjacency_matrix_.index) == list(numeric_df.columns)
        assert list(model.adjacency_matrix_.columns) == list(numeric_df.columns)
        assert set(model.causal_graph_.nodes()) == set(numeric_df.columns)

    def test_adjacency_and_graph_are_consistent(self, fitted_model, numeric_df):
        model, _ = fitted_model
        adjacency = model.adjacency_matrix_
        matrix_edges = {
            (src, dst) for src in adjacency.index for dst in adjacency.columns if adjacency.loc[src, dst] == 1
        }

        assert adjacency.shape == (numeric_df.shape[1], numeric_df.shape[1])
        assert np.all(np.diag(adjacency) == 0)
        assert set(np.unique(adjacency)) <= {0, 1}
        assert isinstance(model.causal_graph_, DAG)
        assert set(model.causal_graph_.nodes()) == set(numeric_df.columns)
        assert nx.is_directed_acyclic_graph(model.causal_graph_)
        assert not any(src == dst for src, dst in model.causal_graph_.edges())
        assert matrix_edges == set(model.causal_graph_.edges())

    def test_final_jacobian_uses_all_scaled_data(self, monkeypatch, numeric_df):
        import torch

        from pgmpy.causal_discovery.gran_dag import _GraNDAGModel

        observed = {}

        def get_jacobian(model, X):
            observed["X"] = X.detach().cpu().numpy()
            return torch.zeros(model.num_vars, model.num_vars, device=X.device, dtype=X.dtype)

        monkeypatch.setattr(_GraNDAGModel, "fit_network", lambda model, X_train, X_val: None)
        monkeypatch.setattr(_GraNDAGModel, "get_jacobian", get_jacobian)

        model = GraNDAG(val_size=0.25, seed=0).fit(numeric_df)

        assert observed["X"].shape[0] == len(numeric_df)
        assert np.allclose(observed["X"], model.scaler_.transform(numeric_df))

    def test_custom_scaler_is_used_and_fitted(self, monkeypatch, numeric_df):
        from pgmpy.causal_discovery.gran_dag import _GraNDAGModel

        scaler = MinMaxScaler()
        monkeypatch.setattr(_GraNDAGModel, "fit_network", lambda model, X_train, X_val: None)
        model = GraNDAG(scaler=scaler, val_size=0, seed=0).fit(numeric_df)

        assert model.scaler_ is scaler

    @pytest.mark.parametrize(
        ("X", "scaler", "match"),
        [
            (pd.DataFrame({"A": [1.0, 2.0]}), None, "at least 2 variables"),
            (pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]}), object(), "scaler must implement"),
        ],
    )
    def test_invalid_fit_inputs_raise(self, X, scaler, match):
        with pytest.raises(ValueError, match=match):
            GraNDAG(scaler=scaler).fit(X)

    @pytest.mark.parametrize(
        ("option", "match"),
        [
            ({"pns_threshold": 0.5}, "PNS is not yet implemented"),
            ({"pruning_cutoff": 0.05}, "CAM pruning is not yet implemented"),
        ],
        ids=["pns", "cam-pruning"],
    )
    def test_unimplemented_graph_processing_raises(self, numeric_df, option, match):
        with pytest.raises(NotImplementedError, match=match):
            GraNDAG(**option).fit(numeric_df)


@requires_torch
class TestGraNDAGModel:
    def _make_model(
        self,
        num_vars=3,
        net=None,
        output_dim=1,
        log_likelihood=None,
        optimizer_params=None,
        max_epochs=1,
        max_subproblems=1,
        patience=2,
        edge_threshold=1e-4,
    ):
        from pgmpy.causal_discovery.gran_dag import _GraNDAGModel

        network_cfg = GraNDAGNetworkConfig(net, output_dim, log_likelihood, None)
        train_cfg = GraNDAGTrainingConfig(
            "rmsprop", optimizer_params, 64, 0.1, max_epochs, 1e-4, patience, max_subproblems, None, 0
        )
        reg_cfg = GraNDAGRegularizationConfig(0.0, 1e-3, 10.0, 0.9, 1e-8, edge_threshold)
        return _GraNDAGModel(num_vars, network_cfg, train_cfg, reg_cfg)

    def test_default_network_and_gaussian_parameters(self):
        import torch
        from torch import nn

        model = self._make_model(num_vars=4)
        layers = list(model.subnets[0])
        linears = [layer for layer in layers if isinstance(layer, nn.Linear)]

        assert [layer.in_features for layer in linears] == [4, 10, 10]
        assert [layer.out_features for layer in linears] == [10, 10, 1]
        assert sum(isinstance(layer, nn.LeakyReLU) for layer in layers) == 2
        assert all(torch.count_nonzero(layer.bias) == 0 for layer in linears)
        assert model(torch.randn(5, 4)).shape == (5, 4, 1)
        assert model.log_var.shape == (4,)
        assert torch.equal(model.log_var.exp(), torch.ones(4))

    def test_get_A_shape_and_diagonal(self):
        import torch

        A = self._make_model(num_vars=4).get_A()
        assert A.shape == (4, 4)
        assert torch.all(A.diagonal() == 0)

    def test_lagrangian_updates_and_mu_growth(self, monkeypatch):
        import torch

        import pgmpy.causal_discovery.gran_dag as gran_dag

        values = iter([0.5, 0.5])
        monkeypatch.setattr(gran_dag, "_dag_constraint", lambda A: A.new_tensor(next(values)))
        model = self._make_model(max_epochs=0, max_subproblems=2)
        model.fit_network(torch.randn(4, 3))

        assert model.lamb == pytest.approx(1e-3)
        assert model.mu == pytest.approx(1e-2)

    def test_outer_loop_stops_at_tolerance(self, monkeypatch):
        import torch

        import pgmpy.causal_discovery.gran_dag as gran_dag

        calls = 0

        def zero_constraint(A):
            nonlocal calls
            calls += 1
            return A.new_tensor(1e-8)

        monkeypatch.setattr(gran_dag, "_dag_constraint", zero_constraint)
        model = self._make_model(max_epochs=0, max_subproblems=5)
        model.fit_network(torch.randn(4, 3))
        assert calls == 1

    @pytest.mark.parametrize(("explicit_lr", "expected"), [(None, [1e-2, 1e-4]), (0.03, [0.03, 0.03])])
    def test_learning_rate_schedule(self, monkeypatch, explicit_lr, expected):
        import torch

        import pgmpy.causal_discovery.gran_dag as gran_dag

        learning_rates = []
        rmsprop = torch.optim.RMSprop

        def record_rmsprop(params, **kwargs):
            learning_rates.append(kwargs["lr"])
            return rmsprop(params, **kwargs)

        monkeypatch.setattr(torch.optim, "RMSprop", record_rmsprop)
        monkeypatch.setattr(gran_dag, "_dag_constraint", lambda A: A.new_tensor(1.0))
        params = None if explicit_lr is None else {"lr": explicit_lr}
        model = self._make_model(optimizer_params=params, max_epochs=0, max_subproblems=2)
        model.fit_network(torch.randn(4, 3))
        assert learning_rates == expected

    def test_online_masking_is_permanent(self):
        import torch

        model = self._make_model(optimizer_params={"lr": 0.0})
        with torch.no_grad():
            for subnet in model.subnets:
                for layer in subnet:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.zero_()

        model.fit_network(torch.randn(8, 3))
        assert torch.count_nonzero(model.adjacency) == 0

        with torch.no_grad():
            for subnet in model.subnets:
                for layer in subnet:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.fill_(1.0)
        assert torch.count_nonzero(model.get_A()) == 0

    def test_validation_patience_resets_for_each_subproblem(self, monkeypatch):
        import torch

        import pgmpy.causal_discovery.gran_dag as gran_dag

        steps = 0
        original_step = torch.optim.RMSprop.step

        def record_step(optimizer, *args, **kwargs):
            nonlocal steps
            steps += 1
            return original_step(optimizer, *args, **kwargs)

        monkeypatch.setattr(torch.optim.RMSprop, "step", record_step)
        monkeypatch.setattr(gran_dag, "_dag_constraint", lambda A: A.sum() * 0 + 1.0)
        model = self._make_model(
            optimizer_params={"lr": 0.0}, max_epochs=3, max_subproblems=2, patience=1, edge_threshold=0.0
        )
        X = torch.randn(8, 3)
        model.fit_network(X, X)
        assert steps == 4

    def test_custom_network_and_likelihood(self):
        import torch
        from torch import nn

        net = nn.Sequential(nn.Linear(3, 5), nn.LeakyReLU(), nn.Linear(5, 2))
        calls = []

        def log_likelihood(X, theta):
            calls.append(theta.shape)
            return -(theta[..., 0] ** 2)

        model = self._make_model(net=net, output_dim=2, log_likelihood=log_likelihood)
        X = torch.randn(6, 3)
        theta = model(X)
        assert theta.shape == (6, 3, 2)
        assert model._compute_log_likelihood(X, theta).shape == (6, 3)
        assert calls == [torch.Size([6, 3, 2])]

    def test_gaussian_likelihood_numerics_and_gradient(self):
        import math

        import torch

        model = self._make_model(num_vars=2)
        X = torch.tensor([[1.0, -1.0], [3.0, 2.0]])
        means = torch.tensor([[[0.5], [-0.5]], [[2.0], [1.0]]])
        with torch.no_grad():
            model.log_var.copy_(torch.tensor([math.log(2.0), math.log(0.5)]))

        actual = model._compute_log_likelihood(X, means)
        log_var = model.log_var.expand_as(X)
        expected = -0.5 * math.log(2 * math.pi) - 0.5 * log_var - (X - means[..., 0]) ** 2 / (2 * log_var.exp())
        assert torch.allclose(actual, expected, atol=1e-7)

        # With zero residuals, changing X does not change the node-specific variance term.
        X_shifted = X + 100
        assert torch.allclose(
            model._compute_log_likelihood(X, X.unsqueeze(-1)),
            model._compute_log_likelihood(X_shifted, X_shifted.unsqueeze(-1)),
        )

        (-actual.sum()).backward()
        assert model.log_var.grad is not None
        assert torch.isfinite(model.log_var.grad).all()

    def test_forward_self_masking(self):
        import torch

        model = self._make_model()
        with torch.no_grad():
            for subnet in model.subnets:
                for layer in subnet:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.fill_(1.0)

        X = torch.ones(5, 3)
        output = model(X)
        X_modified = X.clone()
        X_modified[:, 1] = 999.0
        changed = model(X_modified)

        assert torch.allclose(output[:, 1], changed[:, 1])
        assert not torch.allclose(output[:, 0], changed[:, 0])

        with torch.no_grad():
            model.adjacency[0, 1] = 0
            for layer in model._linears[1]:
                layer.weight.fill_(1.0)
        model.reg_cfg.edge_threshold = 0.0
        model.train_cfg.optimizer_params = {"lr": 0.0}
        model.fit_network(torch.randn(4, 3))
        assert model.adjacency[0, 1] == 0

    def test_get_jacobian_orientation_masks_and_chunking(self):
        import torch
        from torch import nn

        net = nn.Sequential(nn.Linear(3, 1, bias=False))
        model = self._make_model(net=net, log_likelihood=lambda X, theta: theta[..., 0])
        weights = ([0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0])
        with torch.no_grad():
            for subnet, weight in zip(model.subnets, weights):
                subnet[0].weight.copy_(torch.tensor([weight]))
            model.adjacency[2, 1] = 0

        X = torch.randn(5, 3)
        jacobian = model.get_jacobian(X)
        chunked = model.get_jacobian(X, chunk_size=2)

        assert jacobian.shape == (3, 3)
        assert torch.all(jacobian.diagonal() == 0)
        assert jacobian[1, 0].item() == pytest.approx(2.0)
        assert jacobian[0, 1].item() == pytest.approx(3.0)
        assert jacobian[2, 1] == 0
        assert torch.allclose(jacobian, chunked, atol=1e-7)

    @pytest.mark.parametrize(
        ("jacobian", "expected_edges"),
        [
            ([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]], {(0, 1), (1, 2)}),
            ([[0.0, 0.8, 0.0], [0.0, 0.0, 0.6], [0.2, 0.0, 0.0]], {(0, 1), (1, 2)}),
            ([[0.0, 1e-5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], set()),
        ],
        ids=["acyclic", "weakest-cycle-edge", "empty"],
    )
    def test_threshold_to_dag(self, jacobian, expected_edges):
        import torch

        adj = self._make_model()._threshold_to_dag(torch.tensor(jacobian))
        edges = {tuple(edge) for edge in torch.nonzero(adj).tolist()}

        assert edges == expected_edges
        assert set(torch.unique(adj).tolist()) <= {0.0, 1.0}
        assert torch.all(adj.diagonal() == 0)
        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(adj.numpy(), create_using=nx.DiGraph))

    def test_sufficient_progress_preserves_mu_and_max_subproblems(self, monkeypatch):
        import torch

        import pgmpy.causal_discovery.gran_dag as gran_dag

        constraints = iter([0.5, 0.4])
        calls = 0

        def constraint(A):
            nonlocal calls
            calls += 1
            return A.new_tensor(next(constraints))

        monkeypatch.setattr(gran_dag, "_dag_constraint", constraint)
        model = self._make_model(max_epochs=0, max_subproblems=2)
        model.fit_network(torch.randn(4, 3))

        assert calls == 2
        assert model.mu == pytest.approx(1e-3)

    @pytest.mark.parametrize(
        ("net", "output_dim", "match"),
        [
            (lambda nn: nn.Sequential(nn.Conv1d(1, 1, 1)), 1, "learnable layers"),
            (lambda nn: nn.Sequential(nn.Linear(2, 1)), 1, "first Linear layer"),
            (lambda nn: nn.Sequential(nn.Linear(3, 2)), 1, "last Linear layer"),
            (lambda nn: nn.Sequential(nn.ReLU()), 1, "learnable layers"),
        ],
        ids=["unsupported-layer", "wrong-input", "wrong-output", "no-linear"],
    )
    def test_invalid_custom_networks_raise(self, net, output_dim, match):
        from torch import nn

        with pytest.raises(ValueError, match=match):
            self._make_model(net=net(nn), output_dim=output_dim)
