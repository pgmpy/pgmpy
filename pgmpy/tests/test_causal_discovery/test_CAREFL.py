"""Tests for the public CAREFL causal-discovery estimator."""

import importlib

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import CAREFL
from pgmpy.causal_discovery.CAREFL import (
    FlowConfig,
    TrainingConfig,
    _AffineARFlow,
    _CAREFLModel,
    _ConditionerMLP,
)

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)

carefl_module = importlib.import_module(CAREFL.__module__)


@pytest.fixture
def numeric_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((50, 2)), columns=["X", "Y"])


class FakeCAREFLModel:
    """Record public CAREFL orchestration while its private model is a stub."""

    instances = []
    log_prob_outputs = []

    def __init__(self, flow_cfg, train_cfg):
        self.flow_cfg = flow_cfg
        self.train_cfg = train_cfg
        self.train_tensor = None
        self.test_tensor = None
        self.index = len(self.instances)
        self.instances.append(self)

    def fit_network(self, x_train):
        self.train_tensor = x_train.detach().cpu().clone()
        return self

    def eval(self):
        return self

    def log_prob(self, x_test):
        import torch

        self.test_tensor = x_test.detach().cpu().clone()
        if self.index < len(self.log_prob_outputs):
            return torch.as_tensor(
                self.log_prob_outputs[self.index],
                dtype=x_test.dtype,
                device=x_test.device,
            )
        return torch.zeros(x_test.shape[0], dtype=x_test.dtype, device=x_test.device)


def expected_failed_checks(estimator):
    checks = {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y in score.",
        "check_n_features_in_after_fitting": "BaseCausalDiscovery score compatibility limitation.",
        "check_fit2d_1feature": "CAREFL requires exactly two variables.",
    }
    checks.update(
        dict.fromkeys(
            (
                "check_dont_overwrite_parameters",
                "check_positive_only_tag_during_fit",
                "check_estimators_dtypes",
                "check_dtype_object",
                "check_pipeline_consistency",
                "check_estimators_nan_inf",
                "check_estimators_pickle",
                "check_f_contiguous_array_estimator",
                "check_methods_sample_order_invariance",
                "check_methods_subset_invariance",
                "check_dict_unchanged",
                "check_fit2d_predict1d",
            ),
            "The sklearn check fits more than two variables; CAREFL is bivariate-only.",
        )
    )
    return checks


if _check_soft_dependencies("torch", severity="none"):

    @parametrize_with_checks(
        [CAREFL(max_epochs=1, seed=0)],
        expected_failed_checks=expected_failed_checks,
    )
    def test_carefl_compatibility(estimator, check):
        check(estimator)


class TestConditionerMLP:
    @requires_torch
    def test_architecture_and_forward(self):
        import torch

        model = _ConditionerMLP(hidden_dim=8, hidden_layers=3)
        linear_layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]

        assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
            (1, 8),
            (8, 8),
            (8, 8),
            (8, 1),
        ]
        assert model(torch.randn(5, 1)).shape == (5, 1)


class TestAffineARFlow:
    @requires_torch
    def test_round_trip_and_shapes(self):
        import torch

        torch.manual_seed(0)
        flow = _AffineARFlow(hidden_dim=8, hidden_layers=2)
        z = torch.randn(5, 2)
        x = flow(z)
        recovered_z, log_abs_det_dx = flow.inverse(x)
        original_x = torch.randn(5, 2)
        recovered_x = flow(flow.inverse(original_x)[0])

        assert x.shape == (5, 2)
        assert recovered_z.shape == (5, 2)
        assert recovered_x.shape == (5, 2)
        assert log_abs_det_dx.shape == (5,)
        assert torch.allclose(recovered_z, z, atol=1e-6)
        assert torch.allclose(recovered_x, original_x, atol=1e-6)

    @requires_torch
    def test_inverse_log_jacobian_matches_autograd(self):
        import torch

        torch.manual_seed(0)
        flow = _AffineARFlow(hidden_dim=4, hidden_layers=1).double()
        x = torch.randn(2, dtype=torch.float64, requires_grad=True)

        _, analytical_log_det = flow.inverse(x.unsqueeze(0))
        jacobian = torch.autograd.functional.jacobian(
            lambda value: flow.inverse(value.unsqueeze(0))[0].squeeze(0),
            x,
        )
        _, autograd_log_det = torch.linalg.slogdet(jacobian)

        assert torch.allclose(analytical_log_det[0], autograd_log_det, atol=1e-8)


@requires_torch
class TestCAREFLModel:
    @staticmethod
    def make_model(num_flows=2, max_epochs=1):
        return _CAREFLModel(
            FlowConfig(num_flows=num_flows, hidden_dim=4, hidden_layers=1),
            TrainingConfig(
                batch_size=8,
                max_epochs=max_epochs,
                optimizer="adam",
                optimizer_kwargs={"lr": 1e-3, "betas": (0.9, 0.999)},
                scheduler_kwargs={"factor": 0.1},
                seed=0,
            ),
        )

    def test_flow_stack_order_and_log_jacobian(self):
        import torch

        calls = []

        class RecordingFlow(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                calls.append(("forward", self.value, x.clone()))
                return x + self.value

            def inverse(self, x):
                calls.append(("inverse", self.value, x.clone()))
                return x - self.value, x.new_full((x.shape[0],), self.value)

        model = self.make_model(num_flows=3)
        assert len(model.flows) == 3
        model.flows = torch.nn.ModuleList([RecordingFlow(value) for value in (1, 2, 3)])
        z = torch.zeros(2, 2)

        x = model(z)
        recovered_z, log_det = model.inverse(x)

        assert [(operation, value) for operation, value, _ in calls] == [
            ("forward", 1),
            ("forward", 2),
            ("forward", 3),
            ("inverse", 3),
            ("inverse", 2),
            ("inverse", 1),
        ]
        assert [tensor[0, 0].item() for _, _, tensor in calls] == [0, 1, 3, 6, 3, 1]
        assert x.shape == recovered_z.shape == (2, 2)
        assert log_det.shape == (2,)
        assert torch.equal(recovered_z, z)
        assert torch.equal(log_det, torch.full((2,), 6.0))

    def test_log_prob_change_of_variables(self, monkeypatch):
        import torch

        model = self.make_model()
        z = torch.tensor([[0.0, 1.0], [-1.0, 2.0]])
        log_det = torch.tensor([0.5, -0.25])
        monkeypatch.setattr(model, "inverse", lambda x: (z, log_det))

        actual = model.log_prob(torch.zeros(2, 2))
        expected = torch.distributions.Laplace(0.0, 1.0).log_prob(z).sum(dim=1) + log_det

        assert actual.shape == (2,)
        assert torch.allclose(actual, expected)

    def test_fit_network_updates_parameters(self):
        import torch

        torch.manual_seed(0)
        model = self.make_model(num_flows=1, max_epochs=1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        x_train = torch.randn(16, 2)

        result = model.fit_network(x_train)

        assert result is model
        assert any(not torch.equal(previous, current) for previous, current in zip(before, model.parameters()))


class TestOptimizerValidation:
    @requires_torch
    def test_invalid_optimizer_string_raises(self, numeric_df):
        with pytest.raises(ValueError, match="Supported optimizers are"):
            CAREFL(optimizer="rmsprop", max_epochs=1).fit(numeric_df)

    @requires_torch
    @pytest.mark.parametrize(
        ("optimizer", "bad_kwargs", "match"),
        [
            ("adam", {"momentum": 0.9}, "Unknown optimizer_kwargs"),
            ("sgd", {"betas": (0.9, 0.999)}, "Unknown optimizer_kwargs"),
            ("adamw", {"nesterov": True}, "Unknown optimizer_kwargs"),
            ("adam", {"params": [1, 2, 3]}, "params"),
        ],
    )
    def test_invalid_kwargs_raises(self, numeric_df, optimizer, bad_kwargs, match):
        with pytest.raises(ValueError, match=match):
            CAREFL(
                optimizer=optimizer,
                optimizer_kwargs=bad_kwargs,
                max_epochs=1,
            ).fit(numeric_df)


@requires_torch
class TestCAREFLFit:
    @pytest.fixture(autouse=True)
    def fake_model(self, monkeypatch):
        FakeCAREFLModel.instances = []
        FakeCAREFLModel.log_prob_outputs = []
        monkeypatch.setattr(carefl_module, "_CAREFLModel", FakeCAREFLModel)

    def test_two_direction_orchestration_uses_one_shared_split(self):
        data = pd.DataFrame(
            {
                "X": np.arange(30, dtype=float),
                "Y": 1000 + np.arange(30, dtype=float),
            }
        )

        estimator = CAREFL(seed=7).fit(data)
        forward, backward = FakeCAREFLModel.instances

        assert len(FakeCAREFLModel.instances) == 2
        np.testing.assert_array_equal(backward.train_tensor, forward.train_tensor[:, [1, 0]])
        np.testing.assert_array_equal(backward.test_tensor, forward.test_tensor[:, [1, 0]])
        assert set(estimator.models_) == {("X", "Y"), ("Y", "X")}

    @pytest.mark.parametrize(
        (
            "outputs",
            "expected_score",
            "expected_direction",
            "expected_edges",
            "expected_adjacency",
        ),
        [
            (([1.0, 3.0], [0.0, 0.0]), 2.0, ("X", "Y"), {("X", "Y")}, [[0, 1], [0, 0]]),
            (
                ([-2.0, 0.0], [1.0, 3.0]),
                -3.0,
                ("Y", "X"),
                {("Y", "X")},
                [[0, 0], [1, 0]],
            ),
            (([1.0, 3.0], [2.0, 2.0]), 0.0, None, set(), [[0, 0], [0, 0]]),
        ],
    )
    def test_direction_selection(
        self,
        outputs,
        expected_score,
        expected_direction,
        expected_edges,
        expected_adjacency,
    ):
        FakeCAREFLModel.log_prob_outputs = outputs
        data = pd.DataFrame({"X": np.arange(10, dtype=float), "Y": np.arange(10, dtype=float) ** 2})

        estimator = CAREFL(test_size=0.2).fit(data)

        assert estimator.log_likelihoods_ == {
            ("X", "Y"): np.mean(outputs[0]),
            ("Y", "X"): np.mean(outputs[1]),
        }
        assert estimator.causal_score_ == expected_score
        assert estimator.causal_direction_ == expected_direction
        assert set(estimator.causal_graph_.edges()) == expected_edges
        pd.testing.assert_frame_equal(
            estimator.adjacency_matrix_,
            pd.DataFrame(
                expected_adjacency,
                index=["X", "Y"],
                columns=["X", "Y"],
                dtype="int",
            ),
        )

    @pytest.mark.parametrize(
        ("data", "kwargs", "match"),
        [
            (
                pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 3.0], "Z": [3.0, 4.0]}),
                {},
                "exactly two",
            ),
            (
                pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [2.0, 3.0, 4.0]}),
                {},
                "non-constant",
            ),
            (
                pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 3.0]}),
                {"test_size": 1.0},
                "test_size",
            ),
        ],
    )
    def test_carefl_specific_validation(self, data, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CAREFL(**kwargs).fit(data)

    def test_train_config_resolves_defaults_without_mutating_caller_kwargs(self):
        optimizer_kwargs = {"lr": 5e-4}
        scheduler_kwargs = {"patience": 3}
        data = pd.DataFrame({"X": np.arange(10, dtype=float), "Y": np.arange(10, dtype=float) ** 2})

        estimator = CAREFL(
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        ).fit(data)

        assert estimator.train_config_.optimizer_kwargs == {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
        }
        assert estimator.train_config_.scheduler_kwargs == {
            "factor": 0.1,
            "patience": 3,
        }
        assert optimizer_kwargs == {"lr": 5e-4}
        assert scheduler_kwargs == {"patience": 3}

    def test_nonfinite_held_out_likelihood_is_rejected(self):
        FakeCAREFLModel.log_prob_outputs = [[np.nan, 0.0], [0.0, 0.0]]
        data = pd.DataFrame({"X": np.arange(10, dtype=float), "Y": np.arange(10, dtype=float) ** 2})

        with pytest.raises(ValueError, match="finite"):
            CAREFL(test_size=0.2).fit(data)
