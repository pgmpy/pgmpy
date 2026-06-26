"""
Tests for the CASTLE class in pgmpy.causal_discovery.
"""

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.causal_discovery import CASTLE
from pgmpy.causal_discovery.castle import (
    NetworkConfig,
    RegularizationConfig,
    TrainingConfig,
    _CASTLEModel,
    _dag_constraint,
)

requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)


@pytest.fixture
def numeric_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((50, 3)), columns=["A", "B", "C"])


class TestDataclasses:
    def test_dataclasses_are_dataclasses(self):
        assert dataclasses.is_dataclass(NetworkConfig)
        assert dataclasses.is_dataclass(TrainingConfig)
        assert dataclasses.is_dataclass(RegularizationConfig)


@requires_torch
class TestDagConstraint:
    def test_returns_scalar_tensor(self):
        import torch

        W = torch.zeros(4, 4)
        result = _dag_constraint(W)
        assert result.shape == torch.Size([])

    def test_zero_matrix_is_dag(self):
        import torch

        W = torch.zeros(4, 4)
        assert _dag_constraint(W).item() == pytest.approx(0.0, abs=1e-5)

    def test_non_dag_matrix_is_positive(self):
        import torch

        # A cyclic graph: node 0 -> node 1 -> node 0
        W = torch.zeros(4, 4)
        W[0, 1] = 1.0
        W[1, 0] = 1.0
        assert _dag_constraint(W).item() > 0.0


def test_missing_torch_raises_import_error(monkeypatch):
    import pgmpy.causal_discovery.castle as _castle

    def _raise(*args, **kwargs):
        raise ImportError("CASTLE requires PyTorch. Install it with: pip install torch")

    monkeypatch.setattr(_castle, "_check_soft_dependencies", _raise)

    with pytest.raises(ImportError, match="PyTorch"):
        CASTLE()


@requires_torch
class TestCASTLEInit:
    def test_default_params(self):
        est = CASTLE()
        assert est.dag_weight == 1.0
        assert est.sparsity_weight == 5.0
        assert est.dag_penalty == 1.0
        assert est.optimizer == "adam"
        assert est.optimizer_kwargs == {}
        assert est.batch_size == 32
        assert est.hidden_dim == 32
        assert est.edge_threshold == 0.3
        assert est.target_col is None
        assert est.max_epochs == 200
        assert est.min_loss_improvement == 1e-4
        assert est.early_stop_patience == 10
        assert est.scaler is None
        assert est.tensorboard_log_dir is None
        assert est.seed == 42

    def test_get_params_round_trip(self):
        est = CASTLE()
        assert CASTLE(**est.get_params()).get_params() == est.get_params()


@requires_torch
class TestCASTLEFit:
    def test_n_features_in_set_after_fit(self, numeric_df):
        est = CASTLE()
        est.fit(numeric_df)
        assert est.n_features_in_ == numeric_df.shape[1]

    def test_feature_names_in_set_after_fit(self, numeric_df):
        est = CASTLE()
        est.fit(numeric_df)
        assert list(est.feature_names_in_) == list(numeric_df.columns)

    def test_configs_populated(self, numeric_df):
        est = CASTLE(
            hidden_dim=16,
            scaler=None,
            target_col="A",
            batch_size=16,
            max_epochs=5,
            optimizer="sgd",
            lr=0.01,
            momentum=0.9,
            seed=0,
            min_loss_improvement=1e-3,
            early_stop_patience=5,
            tensorboard_log_dir=None,
            dag_weight=2.0,
            sparsity_weight=3.0,
            dag_penalty=0.5,
            edge_threshold=0.2,
        )
        est.fit(numeric_df)

        assert isinstance(est.network_config_, NetworkConfig)
        assert est.network_config_.hidden_dim == 16
        assert est.network_config_.scaler is None
        assert est.network_config_.target_col == "A"

        assert isinstance(est.train_config_, TrainingConfig)
        assert est.train_config_.batch_size == 16
        assert est.train_config_.max_epochs == 5
        assert est.train_config_.optimizer == "sgd"
        assert est.train_config_.optimizer_kwargs == {"lr": 0.01, "momentum": 0.9}
        assert est.train_config_.seed == 0
        assert est.train_config_.min_loss_improvement == 1e-3
        assert est.train_config_.early_stop_patience == 5
        assert est.train_config_.tensorboard_log_dir is None

        assert isinstance(est.reg_config_, RegularizationConfig)
        assert est.reg_config_.dag_weight == 2.0
        assert est.reg_config_.sparsity_weight == 3.0
        assert est.reg_config_.dag_penalty == 0.5
        assert est.reg_config_.edge_threshold == 0.2

    def test_nan_raises(self, numeric_df):
        df_nan = numeric_df.copy()
        df_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            CASTLE().fit(df_nan)

    def test_fit_accepts_dataframe(self, numeric_df):
        CASTLE().fit(numeric_df)

    # --- Group A: input validation ---

    @pytest.mark.parametrize(
        ("df_fn", "kwargs", "match"),
        [
            (lambda df: df[["A"]], {}, "at least 2 columns"),
            (lambda df: df.assign(A=df["A"].astype(str)), {}, "numeric"),
            (lambda df: df, {"target_col": "Z"}, "target_col"),
            (lambda df: df, {"target_col": 99}, "target_col"),
        ],
    )
    def test_invalid_input_raises(self, numeric_df, df_fn, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CASTLE(max_epochs=1, **kwargs).fit(df_fn(numeric_df))

    # --- Group B: structural correctness ---

    def test_all_attributes_set_after_fit(self, numeric_df):
        est = CASTLE(max_epochs=5, seed=0)
        est.fit(numeric_df)
        assert est.causal_graph_ is not None
        assert est.adjacency_matrix_ is not None
        assert est.model_ is not None
        assert est.scaler_ is not None
        assert est.predictor_names_ is not None
        assert est.cols_ is not None

    def test_adjacency_matrix_shape(self, numeric_df):
        est = CASTLE(max_epochs=5, seed=0)
        est.fit(numeric_df)
        d = numeric_df.shape[1]
        assert est.adjacency_matrix_.shape == (d, d)

    def test_causal_graph_valid(self, numeric_df):
        from pgmpy.base import DAG

        est = CASTLE(max_epochs=5, seed=0)
        est.fit(numeric_df)
        assert isinstance(est.causal_graph_, DAG)
        assert set(est.causal_graph_.nodes()) == set(est.cols_)
        assert not any(u == v for u, v in est.causal_graph_.edges())

    @pytest.mark.parametrize(
        "target_col",
        ["A", 0, None],
    )
    def test_target_col_variants_same_cols(self, numeric_df, target_col):
        est = CASTLE(max_epochs=5, seed=0, target_col=target_col)
        est.fit(numeric_df)
        assert est.cols_[0] == "A"

    def test_custom_scaler_used(self, numeric_df):
        from sklearn.preprocessing import StandardScaler

        custom_scaler = StandardScaler()
        est = CASTLE(max_epochs=5, seed=0, scaler=custom_scaler)
        est.fit(numeric_df)
        assert est.scaler_ is custom_scaler


class TestOptimizerValidation:
    def test_invalid_optimizer_string_raises(self):
        with pytest.raises(ValueError, match="Supported optimizers are"):
            CASTLE(optimizer="rmsprop")

    @pytest.mark.parametrize(
        ("optimizer", "bad_kwargs"),
        [
            ("adam", {"momentum": 0.9}),
            ("sgd", {"betas": (0.9, 0.999)}),
            ("adamw", {"nesterov": True}),
        ],
    )
    def test_unknown_kwarg_raises(self, optimizer, bad_kwargs):
        with pytest.raises(ValueError, match="Unknown optimizer_kwargs"):
            CASTLE(optimizer=optimizer, **bad_kwargs)

    @requires_torch
    @pytest.mark.parametrize(
        ("optimizer", "valid_kwargs"),
        [
            ("adam", {"lr": 1e-4, "betas": (0.9, 0.999)}),
            ("sgd", {"lr": 0.01, "momentum": 0.9}),
            ("adamw", {"lr": 5e-4, "weight_decay": 1e-4}),
        ],
    )
    def test_valid_kwargs_accepted(self, optimizer, valid_kwargs):
        CASTLE(optimizer=optimizer, **valid_kwargs)

    def test_params_kwarg_raises(self):
        with pytest.raises(ValueError, match="params"):
            CASTLE(optimizer="adam", params=[1, 2, 3])

    @requires_torch
    def test_case_insensitive_optimizer_name(self):
        CASTLE(optimizer="Adam")
        CASTLE(optimizer="SGD", lr=0.01)
        CASTLE(optimizer="AdamW")


class TestCASTLEModel:
    def _make_model(self, num_inputs=4, hidden_dim=8):
        network_cfg = NetworkConfig(hidden_dim=hidden_dim, scaler=None, target_col=None)
        train_cfg = TrainingConfig(
            batch_size=32,
            max_epochs=1,
            optimizer="adam",
            optimizer_kwargs={},
            seed=None,
            min_loss_improvement=1e-4,
            early_stop_patience=10,
            tensorboard_log_dir=None,
        )
        reg_cfg = RegularizationConfig(dag_weight=1.0, sparsity_weight=5.0, dag_penalty=1.0, edge_threshold=0.3)
        return _CASTLEModel(num_inputs=num_inputs, network_cfg=network_cfg, train_cfg=train_cfg, reg_cfg=reg_cfg)

    # --- get_W ---

    @requires_torch
    def test_get_W_shape(self):
        num_inputs = 4
        model = self._make_model(num_inputs=num_inputs)
        W = model.get_W()
        assert W.shape == (num_inputs, num_inputs)

    @requires_torch
    def test_get_W_diagonal_all_zeros(self):
        import torch

        num_inputs = 4
        model = self._make_model(num_inputs=num_inputs)
        assert torch.all(model.get_W().diagonal() == 0.0)

    @requires_torch
    def test_get_W_non_negative(self):
        num_inputs = 4
        model = self._make_model(num_inputs=num_inputs)
        assert (model.get_W() >= 0.0).all()

    # --- __init__ ---

    @requires_torch
    def test_castle_model_signature(self):
        sig = inspect.signature(_CASTLEModel.__init__)
        params = list(sig.parameters.keys())
        assert "num_inputs" in params
        assert "network_cfg" in params
        assert "train_cfg" in params
        assert "reg_cfg" in params

    @requires_torch
    def test_layer_shapes(self):
        num_inputs, hidden_dim = 4, 8
        model = self._make_model(num_inputs=num_inputs, hidden_dim=hidden_dim)
        assert len(model.input_layers) == num_inputs
        assert len(model.output_layers) == num_inputs
        assert len(model.hidden_layers) == 1
        for k in range(num_inputs):
            assert model.input_layers[k].weight.shape == (hidden_dim, num_inputs)
            assert model.output_layers[k].weight.shape == (1, hidden_dim)
        assert model.hidden_layers[0].weight.shape == (hidden_dim, hidden_dim)

    @requires_torch
    def test_mask_buffers_exist_and_shape(self):
        num_inputs, hidden_dim = 4, 8
        model = self._make_model(num_inputs=num_inputs, hidden_dim=hidden_dim)
        buffer_names = dict(model.named_buffers()).keys()
        for k in range(num_inputs):
            assert f"mask_{k}" in buffer_names
            assert getattr(model, f"mask_{k}").shape == (hidden_dim, num_inputs)

    @requires_torch
    def test_mask_diagonal_column_zeroed(self):
        import torch

        num_inputs, hidden_dim = 4, 8
        model = self._make_model(num_inputs=num_inputs, hidden_dim=hidden_dim)
        for k in range(num_inputs):
            mask = getattr(model, f"mask_{k}")
            assert torch.all(mask[:, k] == 0)
            for j in range(num_inputs):
                if j != k:
                    assert torch.all(mask[:, j] == 1)

    # --- forward ---

    @requires_torch
    def test_output_shapes(self):
        import torch

        B, num_inputs = 10, 4
        model = self._make_model(num_inputs=num_inputs)
        Out, out_0 = model(torch.randn(B, num_inputs))
        assert Out.shape == (B, num_inputs)
        assert out_0.shape == (B, 1)

    @requires_torch
    def test_out_0_matches_first_column(self):
        import torch

        B, num_inputs = 10, 4
        model = self._make_model(num_inputs=num_inputs)
        Out, out_0 = model(torch.randn(B, num_inputs))
        assert torch.allclose(out_0, Out[:, 0:1])

    @requires_torch
    def test_self_masking_in_forward(self):
        import torch

        B, num_inputs = 10, 4
        model = self._make_model(num_inputs=num_inputs)
        X = torch.randn(B, num_inputs)
        Out1, _ = model(X)
        X_modified = X.clone()
        X_modified[:, 2] = 999.0
        Out2, _ = model(X_modified)
        assert torch.allclose(Out1[:, 2], Out2[:, 2])
        assert not torch.allclose(Out1[:, 0], Out2[:, 0])

    @requires_torch
    def test_target_subnetwork_self_masking(self):
        """Sub-network 0 (the target predictor) must not use the target column
        (column 0) as its own input — changing it should not affect Out[:, 0]."""
        import torch

        B, num_inputs = 10, 4
        model = self._make_model(num_inputs=num_inputs)
        X = torch.randn(B, num_inputs)
        Out1, _ = model(X)
        X_modified = X.clone()
        X_modified[:, 0] = 999.0
        Out2, _ = model(X_modified)
        assert torch.allclose(Out1[:, 0], Out2[:, 0])

    @requires_torch
    def test_single_sample_batch(self):
        import torch

        num_inputs = 4
        model = self._make_model(num_inputs=num_inputs)
        Out, out_0 = model(torch.randn(1, num_inputs))
        assert Out.shape == (1, num_inputs)
        assert out_0.shape == (1, 1)


@requires_torch
class TestCASTLEModelTraining:
    @pytest.fixture
    def small_tensor(self):
        import torch

        return torch.randn(20, 4, generator=torch.Generator().manual_seed(0))

    def _make_model(self, num_inputs=4, hidden_dim=8, seed=0, edge_threshold=0.3, max_epochs=5):
        network_cfg = NetworkConfig(hidden_dim=hidden_dim, scaler=None, target_col=None)
        train_cfg = TrainingConfig(
            batch_size=32,
            max_epochs=max_epochs,
            optimizer="adam",
            optimizer_kwargs={},
            seed=seed,
            min_loss_improvement=1e-4,
            early_stop_patience=10,
            tensorboard_log_dir=None,
        )
        reg_cfg = RegularizationConfig(
            dag_weight=1.0, sparsity_weight=5.0, dag_penalty=1.0, edge_threshold=edge_threshold
        )
        return _CASTLEModel(num_inputs=num_inputs, network_cfg=network_cfg, train_cfg=train_cfg, reg_cfg=reg_cfg)

    def test_train_returns_tensor_of_correct_shape(self, small_tensor):
        import torch

        W_final = self._make_model().train(small_tensor)
        assert isinstance(W_final, torch.Tensor)
        assert W_final.shape == (4, 4)

    def test_train_diagonal_is_zero(self, small_tensor):
        import torch

        W_final = self._make_model().train(small_tensor)
        assert torch.all(W_final.diagonal() == 0.0)

    def test_train_no_values_below_threshold(self, small_tensor):
        edge_threshold = 0.3
        W_final = self._make_model(edge_threshold=edge_threshold).train(small_tensor)
        assert not ((W_final > 0.0) & (W_final < edge_threshold)).any()

    def test_train_reproducibility(self, small_tensor):
        import torch

        W1 = self._make_model(seed=42).train(small_tensor)
        W2 = self._make_model(seed=42).train(small_tensor)
        assert torch.equal(W1, W2)
