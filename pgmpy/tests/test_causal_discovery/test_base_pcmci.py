from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery._base_pcmpi import _BasePCMCI


class DummyPCMCI(_BasePCMCI):
    """Concrete PCMCI implementation used for testing."""

    def _initialize_algorithm(self):
        self.initialized_ = True

    def _run_discovery(self, **fit_params):
        self.run_params_ = fit_params

        shape = (self.n_variables_, self.n_variables_, self.tau_max + 1)

        return {
            "graph": np.full(shape, "", dtype="<U3"),
            "p_matrix": np.zeros(shape),
            "val_matrix": np.ones(shape),
        }


class TestBasePCMCI:
    """Tests for the shared PCMCI implementation."""

    @pytest.fixture
    def model(self):
        return DummyPCMCI(cond_ind_test=MagicMock())

    @pytest.fixture
    def data(self):
        index = pd.MultiIndex.from_product(
            [["s1", "s2"], range(5)],
            names=["series_id", "time"],
        )

        return pd.DataFrame(
            {
                "X": np.arange(10, dtype=float),
                "Y": np.arange(10, dtype=float) * 2,
            },
            index=index,
        )

    # ------------------------------------------------------------------
    # Initialization and parameter validation
    # ------------------------------------------------------------------

    def test_initialization(self, model):
        assert model.tau_min == 0
        assert model.tau_max == 1
        assert model.pc_alpha == 0.05
        assert model.fdr_method == "fdr_bh"
        assert getattr(model, "pcmci_", None) is None

    @pytest.mark.parametrize(
        "pc_alpha",
        [
            0,
            1,
            -0.1,
            1.1,
            True,
            np.bool_(False),
            "0.05",
            1 + 2j,
        ],
    )
    def test_invalid_pc_alpha(self, pc_alpha):
        model = DummyPCMCI(
            cond_ind_test=MagicMock(),
            pc_alpha=pc_alpha,
        )

        with pytest.raises((TypeError, ValueError)):
            model._validate_parameters()

    @pytest.mark.parametrize(
        "pc_alpha",
        [0.01, 0.05, 0.1, None, [0.01, 0.05], (0.01, 0.05)],
    )
    def test_valid_pc_alpha(self, pc_alpha):
        model = DummyPCMCI(
            cond_ind_test=MagicMock(),
            pc_alpha=pc_alpha,
        )

        model._validate_parameters()

    def test_invalid_tau_parameters(self):
        invalid_parameters = [
            {"tau_min": -1, "tau_max": 1},
            {"tau_min": 0, "tau_max": -1},
            {"tau_min": 2, "tau_max": 1},
            {"tau_min": 0.5, "tau_max": 1},
            {"tau_min": 0, "tau_max": 1.5},
            {"tau_min": True, "tau_max": 1},
        ]

        for params in invalid_parameters:
            model = DummyPCMCI(**params)

            with pytest.raises((TypeError, ValueError)):
                model._validate_parameters()

    # ------------------------------------------------------------------
    # Input validation and data preparation
    # ------------------------------------------------------------------

    def test_check_fit_data(self, model, data):
        model._check_fit_data(data)

        assert model.variable_names_ is None
        assert model.series_data_ is None

    def test_prepare_data(self, model, data):
        model._check_fit_data(data)
        model._prepare_data(data)

        assert model.n_variables_ == 2
        assert model.n_series_ == 2
        assert model.variable_names_ == ["X", "Y"]
        assert model.series_lengths_ == {"s1": 5, "s2": 5}

        assert set(model.series_data_) == {"s1", "s2"}
        assert model.series_data_["s1"].shape == (5, 2)
        assert model.series_data_["s2"].shape == (5, 2)

    def test_check_fit_data_invalid_input(self, model):
        with pytest.raises((TypeError, ValueError)):
            model._check_fit_data(np.array([[1, 2], [3, 4]]))

    def test_check_fit_data_requires_multiindex(self, model):
        data = pd.DataFrame(
            {"X": [1.0, 2.0, 3.0]},
            index=pd.RangeIndex(3),
        )

        with pytest.raises((TypeError, ValueError)):
            model._check_fit_data(data)

    def test_check_fit_data_requires_two_index_levels(self, model):
        index = pd.Index(["s1", "s1", "s1"])

        data = pd.DataFrame({"X": [1.0, 2.0, 3.0]}, index=index)

        with pytest.raises((TypeError, ValueError)):
            model._check_fit_data(data)

    # def test_prepare_data_rejects_missing_values(self, model):
    #     index = pd.MultiIndex.from_product(
    #         [["s1"], range(3)],
    #         names=["series_id", "time"],
    #     )
    #     data = pd.DataFrame({"X": [1.0, np.nan, 3.0]}, index=index)

    #     model._check_fit_data(data)

    #     with pytest.raises(ValueError):
    #         model._prepare_data(data)

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0, np.nan, 4.0],
            [1.0, 2.0, np.inf, 4.0],
            [1.0, 2.0, -np.inf, 4.0],
        ],
        ids=["nan", "positive-infinity", "negative-infinity"],
    )
    def test_check_fit_data_rejects_non_finite_values(self, model, values):
        index = pd.MultiIndex.from_product(
            [["s1"], range(4)],
            names=["series_id", "time"],
        )
        data = pd.DataFrame({"X": values, "Y": [1.0, 2.0, 3.0, 4.0]}, index=index)

        with pytest.raises(ValueError):
            model._check_fit_data(data)

    def test_prepare_data_rejects_duplicate_timestamps(self, model):
        index = pd.MultiIndex.from_tuples(
            [("s1", 0), ("s1", 1), ("s1", 1)],
            names=["series_id", "time"],
        )
        data = pd.DataFrame({"X": [1.0, 2.0, 3.0]}, index=index)

        with pytest.raises(ValueError):
            model._check_fit_data(data)
            model._prepare_data(data)

    def test_prepare_data_rejects_series_too_short(self):
        model = DummyPCMCI(
            cond_ind_test=MagicMock(),
            tau_max=3,
        )

        index = pd.MultiIndex.from_product(
            [["s1"], range(3)],
            names=["series_id", "time"],
        )
        data = pd.DataFrame({"X": [1.0, 2.0, 3.0]}, index=index)

        model._check_fit_data(data)

        with pytest.raises(ValueError):
            model._prepare_data(data)

    # ------------------------------------------------------------------
    # Link assumptions
    # ------------------------------------------------------------------

    def test_validate_link_assumptions(self, model, data):
        model._check_fit_data(data)
        model._prepare_data(data)

        model.link_assumptions = {
            0: {(1, -1): "-->"},
        }

        model._validate_link_assumptions()

        assert model.link_assumptions_ == {
            0: {(1, -1): "-->"},
        }

    def test_link_assumption_alias_normalization(self, model, data):
        model._check_fit_data(data)
        model._prepare_data(data)

        model.link_assumptions = {
            0: {(1, -1): "->"},
        }

        model._validate_link_assumptions()

        assert model.link_assumptions_[0][(1, -1)] == "-->"

    def test_invalid_link_assumption_variable_index(self, model, data):
        model._check_fit_data(data)
        model._prepare_data(data)

        model.link_assumptions = {
            2: {(0, -1): "-->"},
        }

        with pytest.raises(ValueError):
            model._validate_link_assumptions()

    # ------------------------------------------------------------------
    # Discovery results
    # ------------------------------------------------------------------

    def test_process_discovery_results(self, model):
        model.n_variables_ = 2

        shape = (2, 2, model.tau_max + 1)

        results = {
            "graph": np.full(shape, "", dtype="<U3"),
            "p_matrix": np.zeros(shape),
            "val_matrix": np.ones(shape),
        }

        model._process_discovery_results(results)

        assert model.graph_.shape == shape
        assert model.p_matrix_.shape == shape
        assert model.val_matrix_.shape == shape

        np.testing.assert_array_equal(model.p_matrix_, results["p_matrix"])
        np.testing.assert_array_equal(model.val_matrix_, results["val_matrix"])

    def test_process_discovery_results_missing_key(self, model):
        model.n_variables_ = 2

        with pytest.raises(ValueError, match="missing required keys"):
            model._process_discovery_results({"graph": np.zeros((2, 2, 2))})

    def test_process_discovery_results_invalid_shape(self, model):
        model.n_variables_ = 2

        results = {
            "graph": np.zeros((2, 2, 1)),
            "p_matrix": np.zeros((2, 2, 1)),
            "val_matrix": np.zeros((2, 2, 1)),
        }

        with pytest.raises(ValueError):
            model._process_discovery_results(results)

    def test_process_discovery_results_invalid_type(self, model):
        with pytest.raises((TypeError, ValueError)):
            model._process_discovery_results(None)

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------

    def test_fit(self, model, data):
        model._check_fit_data(data)
        model._fit(data)

        assert model.initialized_ is True
        assert model.graph_ is not None
        assert model.p_matrix_ is not None
        assert model.val_matrix_ is not None

        assert model.graph_.shape == (2, 2, model.tau_max + 1)

    def test_fit_passes_fit_parameters_to_discovery(self, model, data):
        model._check_fit_data(data)

        model._fit(data, max_conds_dim=2)

        assert model.run_params_ == {"max_conds_dim": 2}

    def test_reset_fit_state(self, model, data):
        model._check_fit_data(data)
        model._fit(data)

        model._reset_fit_state()

        assert model.series_ids_ is None
        assert model.variable_names_ is None
        assert model.n_variables_ is None
        assert model.n_observations_ is None
        assert model.n_series_ is None
        assert model.series_data_ is None
        assert model.series_lengths_ is None
        assert model.X_ is None
        assert model.ci_test_ is None
        assert model.tigramite_ci_test_ is None
        assert model.link_assumptions_ is None
        assert model.results_ is None
        assert model.graph_ is None
        assert model.p_matrix_ is None
        assert model.val_matrix_ is None
        assert model.pcmci_ is None
