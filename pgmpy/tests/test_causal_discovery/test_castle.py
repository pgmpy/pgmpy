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
        assert est.optimizer is None
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
            optimizer=None,
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
        assert est.train_config_.optimizer is None
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


class TestCASTLEModel:
    @requires_torch
    def test_castle_model_signature(self):
        sig = inspect.signature(_CASTLEModel.__init__)
        params = list(sig.parameters.keys())
        assert "num_inputs" in params
        assert "network_cfg" in params
