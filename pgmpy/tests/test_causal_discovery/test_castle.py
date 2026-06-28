"""
Tests for the CASTLE class in pgmpy.causal_discovery.
"""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.utils.estimator_checks import parametrize_with_checks

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


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
        "check_fit2d_1feature": "CASTLE requires at least one target and one feature column.",
    }


if _check_soft_dependencies("torch", severity="none"):

    @parametrize_with_checks(
        [CASTLE(max_epochs=1, seed=0)],
        expected_failed_checks=expected_failed_checks,
    )
    def test_castle_compatibility(estimator, check):
        check(estimator)


@pytest.fixture
def numeric_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((50, 3)), columns=["A", "B", "C"])


@requires_torch
def test_dag_constraint_behavior():
    import torch

    # Zero matrix (DAG) should return a scalar tensor with value ~0.0
    W_zero = torch.zeros(4, 4)
    res_zero = _dag_constraint(W_zero)
    assert res_zero.shape == torch.Size([])
    assert res_zero.item() == pytest.approx(0.0, abs=1e-5)

    # Cyclic graph (node 0 -> node 1 -> node 0) should be positive
    W_cyclic = torch.zeros(4, 4)
    W_cyclic[0, 1] = 1.0
    W_cyclic[1, 0] = 1.0
    assert _dag_constraint(W_cyclic).item() > 0.0


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
            optimizer_kwargs={"lr": 0.01, "momentum": 0.9},
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

    # --- Group A: input validation ---

    @pytest.mark.parametrize(
        ("df_fn", "kwargs", "match"),
        [
            (lambda df: df[["A"]], {}, "at least 2 columns"),
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

    def test_causal_graph_creation(self, numeric_df):
        from pgmpy.base import DAG

        est = CASTLE(max_epochs=5, seed=0)
        est.fit(numeric_df)
        d = numeric_df.shape[1]

        # Test adjacency matrix shape
        assert est.adjacency_matrix_.shape == (d, d)

        # Test resulting DAG graph
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
        from sklearn.preprocessing import MinMaxScaler

        custom_scaler = MinMaxScaler()
        est = CASTLE(max_epochs=5, seed=0, scaler=custom_scaler)
        est.fit(numeric_df)
        assert est.scaler_ is custom_scaler


class TestOptimizerValidation:
    @requires_torch
    def test_invalid_optimizer_string_raises(self, numeric_df):
        with pytest.raises(ValueError, match="Supported optimizers are"):
            CASTLE(optimizer="rmsprop", max_epochs=1).fit(numeric_df)

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
            CASTLE(optimizer=optimizer, optimizer_kwargs=bad_kwargs, max_epochs=1).fit(numeric_df)


class TestCASTLEModel:
    def _make_model(self, num_inputs=4, hidden_dim=8, seed=None, edge_threshold=0.3, max_epochs=1):
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

    # --- get_W ---

    @requires_torch
    def test_get_W_correctness(self):
        import torch

        num_inputs = 4
        model = self._make_model(num_inputs=num_inputs)
        W = model.get_W()

        assert W.shape == (num_inputs, num_inputs)
        assert torch.all(W.diagonal() == 0.0)
        assert (W >= 0.0).all()

    # --- __init__ ---

    @requires_torch
    def test_mask_buffers_correctness(self):
        import torch

        num_inputs, hidden_dim = 4, 8
        model = self._make_model(num_inputs=num_inputs, hidden_dim=hidden_dim)
        buffer_names = dict(model.named_buffers()).keys()

        for k in range(num_inputs):
            assert f"mask_{k}" in buffer_names
            mask = getattr(model, f"mask_{k}")
            assert mask.shape == (hidden_dim, num_inputs)
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

    # --- train ---

    @pytest.fixture
    def small_tensor(self):
        import torch

        return torch.randn(20, 4, generator=torch.Generator().manual_seed(0))

    @requires_torch
    def test_train_output_correctness(self, small_tensor):
        import torch

        edge_threshold = 0.3
        # Use max_epochs=5 and seed=0 to replicate the old behavior of TestCASTLEModelTraining
        W_final = self._make_model(edge_threshold=edge_threshold, max_epochs=5, seed=0).train(small_tensor)

        assert isinstance(W_final, torch.Tensor)
        assert W_final.shape == (4, 4)
        assert torch.all(W_final.diagonal() == 0.0)
        assert not ((W_final > 0.0) & (W_final < edge_threshold)).any()

    @requires_torch
    def test_train_reproducibility(self, small_tensor):
        import torch

        W1 = self._make_model(seed=42, max_epochs=5).train(small_tensor)
        W2 = self._make_model(seed=42, max_epochs=5).train(small_tensor)
        assert torch.equal(W1, W2)
