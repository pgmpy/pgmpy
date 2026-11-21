from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery._base_pcmpi import _BasePCMCI


class DummyPCMCI(_BasePCMCI):
    """Concrete implementation for testing."""

    def _initialize_algorithm(self):
        pass

    def _run_discovery(self):
        return {}


class TestBasePCMCI:
    """Tests for _BasePCMCI."""

    @pytest.fixture
    def model(self):
        return DummyPCMCI(cond_ind_test=MagicMock())

    @pytest.fixture
    def data(self):
        index = pd.MultiIndex.from_product(
            [["s1", "s2"], range(5)],
            names=["series", "time"],
        )
        return pd.DataFrame(
            {
                "X": np.arange(10, dtype=float),
                "Y": np.arange(10, dtype=float) * 2,
            },
            index=index,
        )

    def test_initialization(self, model):
        assert model.tau_min == 0
        assert model.tau_max == 1
        assert model.pc_alpha == 0.05
        assert model.pcmci_ is None

    def test_invalid_pc_alpha(self):
        with pytest.raises(ValueError):
            DummyPCMCI(cond_ind_test=MagicMock(), pc_alpha=1.5)

    def test_prepare_time_series_data(self, model, data):
        model._validate_time_series_data(data)
        model._prepare_time_series_data(data)
        model._validate_series_lengths()

        assert model.n_variables_ == 2
        assert model.n_series_ == 2
        assert model.variable_names_ == ["X", "Y"]
        assert model.series_lengths_ == {"s1": 5, "s2": 5}

    def test_validate_time_series_data_invalid_input(self, model):
        with pytest.raises((TypeError, ValueError)):
            model._validate_time_series_data(np.array([[1, 2], [3, 4]]))

    def test_validate_time_series_data_missing_values(self, model):
        index = pd.MultiIndex.from_product([["s1"], range(3)])
        data = pd.DataFrame({"X": [1.0, np.nan, 3.0]}, index=index)

        with pytest.raises(ValueError, match="missing"):
            model._validate_time_series_data(data)

    def test_wrap_ci_test(self, model):
        model._wrap_ci_test_for_tigramite()

        assert model.ci_test is not model.cond_ind_test
        assert model.tigramite_ci_test_ is model.ci_test

    def test_wrap_ci_test_without_test(self, model):
        model.cond_ind_test = None

        with pytest.raises(ValueError, match="cond_ind_test"):
            model._wrap_ci_test_for_tigramite()

    def test_link_assumptions(self, model):
        model.n_variables_ = 2
        assumptions = {0: {(1, -1): "->"}}
        model.link_assumptions = assumptions

        assert model._set_link_assumptions() == assumptions

    def test_invalid_link_assumptions(self, model):
        model.n_variables_ = 2
        model.link_assumptions = {2: {(0, -1): "->"}}

        with pytest.raises(ValueError):
            model._set_link_assumptions()

    def test_process_discovery_results(self, model):
        model.n_variables_ = 2
        shape = (2, 2, model.tau_max + 1)

        results = {
            "graph": np.full(shape, "", dtype="<U2"),
            "p_matrix": np.zeros(shape),
            "val_matrix": np.ones(shape),
        }

        model._process_discovery_results(results)

        assert model.graph_.shape == shape
        assert model.p_matrix_.shape == shape
        assert model.val_matrix_.shape == shape

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

        with pytest.raises(ValueError, match="expected"):
            model._process_discovery_results(results)
