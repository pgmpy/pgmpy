"""
Tests for the GraNDAG class in pgmpy.causal_discovery.
"""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

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
    @pytest.fixture
    def numeric_df(self):
        import pandas as pd

        return pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [2.0, 4.0, 6.0, 8.0, 10.0]})

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

    def test_training_without_validation(self):
        import torch

        model = self._make_model(max_epochs=1, max_subproblems=1)
        model.fit_network(torch.randn(8, 3), None)
        assert hasattr(model, "lamb")

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
