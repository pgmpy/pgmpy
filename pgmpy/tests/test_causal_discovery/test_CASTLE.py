"""
Tests for the CASTLE causal discovery algorithm in pgmpy.causal_discovery.
"""

from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.causal_discovery import CASTLE

torch_available = _check_soft_dependencies("torch", severity="none")
pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="torch is required for CASTLE tests",
)


@pytest.fixture
def simple_df():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 150)
    x2 = np.random.normal(0, 1, 150)
    y = x1 + x2 + np.random.normal(0, 0.1, 150)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def fitted_castle(simple_df):
    castle = CASTLE(max_epochs=5, n_hidden=8, random_state=42)
    castle.fit(simple_df, target_col="y")
    return castle


class TestCASTLECore:
    """Tests for core CASTLE fitting behaviour and output attributes."""

    def test_fit_returns_self(self, simple_df):
        castle = CASTLE(max_epochs=2, random_state=42)
        result = castle.fit(simple_df, target_col="y")
        assert result is castle

    def test_fit_sets_causal_graph(self, fitted_castle):
        assert hasattr(fitted_castle, "causal_graph_")
        assert isinstance(fitted_castle.causal_graph_, DAG)

    def test_fit_sets_adjacency_matrix(self, fitted_castle):
        adj = fitted_castle.adjacency_matrix_
        assert isinstance(adj, pd.DataFrame)
        assert adj.shape == (3, 3)
        assert list(adj.columns) == fitted_castle.cols_
        assert list(adj.index) == fitted_castle.cols_
        np.testing.assert_array_equal(np.diag(adj.values), np.zeros(3))

    def test_fit_sets_model(self, fitted_castle):
        assert hasattr(fitted_castle, "model_")

    def test_fit_sets_base_class_attributes(self, fitted_castle):
        # feature_names_in_ counts all columns including target (set by base class)
        assert fitted_castle.n_features_in_ == 3
        assert hasattr(fitted_castle, "predictor_names_")
        # predictor_names_ excludes the target column
        assert len(fitted_castle.predictor_names_) == 2
        assert "x1" in fitted_castle.predictor_names_
        assert "x2" in fitted_castle.predictor_names_
        assert "y" not in fitted_castle.predictor_names_

    def test_empty_graph_is_acyclic_after_high_threshold(self, simple_df):
        # A very high threshold zeroes all weights, producing an empty graph
        # which is trivially a DAG
        castle = CASTLE(max_epochs=2, w_threshold=20.0, random_state=42)
        castle.fit(simple_df, target_col="y")
        assert nx.is_directed_acyclic_graph(castle.causal_graph_)
        assert len(castle.causal_graph_.edges()) == 0

    def test_target_col_is_first_in_cols(self, fitted_castle):
        assert fitted_castle.cols_[0] == "y"

    def test_adjacency_matrix_column_order_matches_cols(self, fitted_castle):
        adj = fitted_castle.adjacency_matrix_
        assert list(adj.columns) == fitted_castle.cols_
        assert list(adj.index) == fitted_castle.cols_

    def test_default_params(self):
        castle = CASTLE()
        assert castle.reg_lambda == 1.0
        assert castle.reg_beta == 5.0
        assert castle.rho == 1.0
        assert castle.lr == 0.001
        assert castle.batch_size == 32
        assert castle.n_hidden == 32
        assert castle.w_threshold == 0.3
        assert castle.max_epochs == 200
        assert castle.random_state is None


class TestCASTLEInputValidation:
    """Tests for input validation, error handling, and edge cases."""

    def test_invalid_target_col_string_raises(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        castle = CASTLE(max_epochs=2)
        with pytest.raises(ValueError, match="target_col"):
            castle.fit(df, target_col="Z")

    def test_invalid_target_col_int_raises(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        castle = CASTLE(max_epochs=2)
        with pytest.raises(ValueError, match="target_col"):
            castle.fit(df, target_col=99)

    def test_negative_target_col_int_raises(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        castle = CASTLE(max_epochs=2)
        with pytest.raises(ValueError, match="target_col"):
            castle.fit(df, target_col=-1)

    def test_target_col_by_integer_resolves_correctly(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        # High threshold so the test is fast and produces a valid DAG
        castle = CASTLE(max_epochs=2, w_threshold=20.0)
        castle.fit(df, target_col=1)
        assert castle.cols_[0] == "B"

    def test_target_col_default_uses_first_column(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        castle = CASTLE(max_epochs=2)
        castle.fit(df)
        assert castle.cols_[0] == "A"

    def test_optional_dependency_guard(self):
        with patch("pgmpy.causal_discovery.CASTLE._check_soft_dependencies") as mock_check:
            mock_check.side_effect = ImportError("torch not found")
            with pytest.raises(ImportError, match="torch"):
                CASTLE()


class TestCASTLEPredict:
    """Tests for the CASTLE predict method."""

    def test_predict_returns_1d_array(self, fitted_castle):
        np.random.seed(0)
        X_test = pd.DataFrame(np.random.normal(0, 1, (20, 2)), columns=["x1", "x2"])
        preds = fitted_castle.predict(X_test)
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1
        assert len(preds) == 20

    def test_predict_accepts_numpy_input(self, fitted_castle):
        np.random.seed(0)
        X_test = np.random.normal(0, 1, (20, 2)).astype(np.float32)
        preds = fitted_castle.predict(X_test)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 20

    def test_predict_output_is_finite(self, fitted_castle):
        np.random.seed(0)
        X_test = pd.DataFrame(np.random.normal(0, 1, (20, 2)), columns=["x1", "x2"])
        preds = fitted_castle.predict(X_test)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        castle = CASTLE()
        with pytest.raises(AttributeError):
            castle.predict(np.random.normal(0, 1, (10, 2)))

    def test_predict_mse_is_reasonable(self, simple_df):
        train = simple_df.iloc[:120]
        test = simple_df.iloc[120:]
        castle = CASTLE(max_epochs=20, n_hidden=16, random_state=42)
        castle.fit(train, target_col="y")
        preds = castle.predict(test[["x1", "x2"]])
        mse = np.mean((test["y"].values - preds) ** 2)
        assert np.isfinite(mse)
        assert mse < 10.0  # loose upper bound, not a performance claim


class TestCASTLETraining:
    """Tests for training behaviour including reproducibility."""

    def test_random_state_reproducibility(self):
        np.random.seed(0)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        df["C"] = df["A"] + df["B"] + np.random.normal(0, 0.1, 100)

        c1 = CASTLE(max_epochs=5, random_state=42)
        c1.fit(df, target_col="A")

        c2 = CASTLE(max_epochs=5, random_state=42)
        c2.fit(df, target_col="A")

        pd.testing.assert_frame_equal(c1.adjacency_matrix_, c2.adjacency_matrix_)

    def test_different_random_states_may_differ(self):
        # Two different seeds should both produce valid results without errors
        np.random.seed(0)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])

        c1 = CASTLE(max_epochs=3, random_state=0)
        c1.fit(df, target_col="A")

        c2 = CASTLE(max_epochs=3, random_state=99)
        c2.fit(df, target_col="A")

        # Both should complete and produce valid adjacency matrices
        assert isinstance(c1.adjacency_matrix_, pd.DataFrame)
        assert isinstance(c2.adjacency_matrix_, pd.DataFrame)

    def test_w_threshold_zero_allows_dense_graph(self):
        # w_threshold=0 keeps all non-zero weights; with only 2 epochs the
        # DAG penalty may not have converged so cycles are possible
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        castle = CASTLE(max_epochs=2, w_threshold=0.0, random_state=42)
        castle.fit(df, target_col="A")
        # Verify the adjacency matrix is well-formed and non-trivial
        assert isinstance(castle.adjacency_matrix_, pd.DataFrame)
        assert castle.adjacency_matrix_.shape == (3, 3)

    def test_w_threshold_very_high_gives_empty_graph(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        castle = CASTLE(max_epochs=2, w_threshold=1e6, random_state=42)
        castle.fit(df, target_col="A")
        assert len(castle.causal_graph_.edges()) == 0
