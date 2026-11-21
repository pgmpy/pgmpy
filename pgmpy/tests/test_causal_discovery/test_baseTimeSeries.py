import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery._baseTimeSeries import (
    BaseTimeSeriesCausalDiscovery,
)


class DummyTimeSeriesEstimator(BaseTimeSeriesCausalDiscovery):
    """Concrete estimator used to test the time-series base class."""

    def __init__(self, tau_min=0, tau_max=1, link_assumptions=None):
        super().__init__(
            tau_min=tau_min,
            tau_max=tau_max,
            cond_ind_test=object(),
            link_assumptions=link_assumptions,
        )
        self.hook_calls = []

    def _initialize_algorithm(self):
        self.hook_calls.append("initialize")

    def _run_discovery(self, **fit_params):
        self.hook_calls.append("run")
        self.fit_params_ = fit_params

        return {
            "graph": "synthetic-link",
            "p_matrix": np.zeros((2, 2, self.tau_max + 1)),
            "val_matrix": np.zeros((2, 2, self.tau_max + 1)),
        }

    def _process_discovery_results(self, results):
        self.hook_calls.append("process")
        self.results_ = results
        self.graph_ = results["graph"]
        self.p_matrix_ = results["p_matrix"]
        self.val_matrix_ = results["val_matrix"]


def make_data(values=None, series_ids=("series_1",), times=range(10)):
    """Create a DataFrame with the required two-level time-series index."""
    index = pd.MultiIndex.from_product(
        [series_ids, times],
        names=["series_id", "time"],
    )

    if values is None:
        values = {
            "X": np.arange(len(index), dtype=float),
            "Y": np.arange(len(index), dtype=float) * 2,
        }

    return pd.DataFrame(values, index=index)


@pytest.fixture
def estimator():
    return DummyTimeSeriesEstimator()


@pytest.fixture
def data():
    """Generate a time series where X[t-1] influences Y[t]."""
    rng = np.random.default_rng(42)
    n_observations = 500

    x = np.zeros(n_observations)
    y = np.zeros(n_observations)

    for t in range(1, n_observations):
        x[t] = 0.7 * x[t - 1] + rng.normal()
        y[t] = 0.8 * x[t - 1] + 0.3 * y[t - 1] + rng.normal(scale=0.5)

    return make_data(
        values={"X": x, "Y": y},
        times=range(n_observations),
    )


class TestBaseTimeSeriesCausalDiscovery:
    def test_fit_runs_discovery_pipeline(self, estimator, data):
        result = estimator.fit(data)

        assert result is estimator
        assert estimator.hook_calls == [
            "initialize",
            "run",
            "process",
        ]

        assert estimator.series_ids_.tolist() == ["series_1"]
        assert estimator.variable_names_ == ["X", "Y"]
        assert estimator.n_variables_ == 2
        assert estimator.n_observations_ == len(data)
        assert estimator.n_series_ == 1
        assert estimator.series_lengths_ == {"series_1": len(data)}

        np.testing.assert_array_equal(
            estimator.series_data_["series_1"].to_numpy(),
            data.to_numpy(),
        )

        assert estimator.graph_ == "synthetic-link"
        assert estimator.results_ is not None

    def test_fit_passes_extra_parameters_to_discovery(self, estimator, data):
        estimator.fit(data, custom_parameter="test")

        assert estimator.fit_params_ == {"custom_parameter": "test"}

    def test_fit_rejects_non_dataframe_input(self, estimator):
        values = np.random.default_rng(0).normal(size=(10, 2))

        with pytest.raises((TypeError, ValueError)):
            estimator.fit(values)

    @pytest.mark.parametrize(
        "index",
        [
            pd.RangeIndex(10),
            pd.Index(range(10), name="time"),
            pd.MultiIndex.from_product(
                [["series_1"], ["a", "b"] * 5],
                names=["series_id", "time"],
            ),
        ],
        ids=[
            "single-level-index",
            "named-single-level-index",
            "duplicate-time-values",
        ],
    )
    def test_fit_rejects_invalid_index(self, estimator, index):
        data = pd.DataFrame(
            {"X": np.arange(10), "Y": np.arange(10)},
            index=index,
        )

        with pytest.raises((TypeError, ValueError)):
            estimator.fit(data)

    def test_fit_rejects_duplicate_variable_names(self, estimator):
        data = pd.DataFrame(
            np.arange(20).reshape(10, 2),
            columns=["X", "X"],
            index=pd.MultiIndex.from_product(
                [["series_1"], range(10)],
                names=["series_id", "time"],
            ),
        )

        with pytest.raises(ValueError):
            estimator.fit(data)

    def test_fit_rejects_non_string_variable_names(self, estimator):
        data = make_data(values={"X": np.arange(10), 1: np.arange(10)})

        with pytest.raises((TypeError, ValueError)):
            estimator.fit(data)

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0, np.nan, 4.0],
            [1.0, 2.0, np.inf, 4.0],
            [1.0, 2.0, -np.inf, 4.0],
        ],
        ids=["nan", "positive-infinity", "negative-infinity"],
    )
    def test_fit_rejects_non_finite_data(self, estimator, values):
        data = make_data(
            values={
                "X": values,
                "Y": [1.0, 2.0, 3.0, 4.0],
            },
            times=range(4),
        )

        with pytest.raises(ValueError):
            estimator.fit(data)

    def test_fit_rejects_non_numeric_data(self, estimator):
        data = make_data(
            values={
                "X": [1.0, 2.0, 3.0, 4.0],
                "Y": ["a", "b", "c", "d"],
            },
            times=range(4),
        )

        with pytest.raises((TypeError, ValueError)):
            estimator.fit(data)

    def test_fit_rejects_complex_data(self, estimator):
        data = make_data(
            values={
                "X": np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]),
                "Y": [1.0, 2.0, 3.0, 4.0],
            },
            times=range(4),
        )

        with pytest.raises((TypeError, ValueError)):
            estimator.fit(data)

    def test_fit_rejects_series_shorter_than_max_lag(self, estimator):
        estimator.tau_max = 3
        data = make_data(
            values={
                "X": [1.0, 2.0, 3.0],
                "Y": [4.0, 5.0, 6.0],
            },
            times=range(3),
        )

        with pytest.raises(ValueError):
            estimator.fit(data)

    def test_fit_sorts_unsorted_time_index(self, estimator):
        index = pd.MultiIndex.from_tuples(
            [
                ("series_1", 2),
                ("series_1", 0),
                ("series_1", 1),
                ("series_1", 3),
            ],
            names=["series_id", "time"],
        )
        data = pd.DataFrame(
            {
                "X": [2.0, 0.0, 1.0, 3.0],
                "Y": [12.0, 10.0, 11.0, 13.0],
            },
            index=index,
        )

        with pytest.warns(UserWarning):
            estimator.fit(data)

        series = estimator.series_data_["series_1"]

        assert series.index.is_monotonic_increasing
        np.testing.assert_array_equal(series["X"], [0.0, 1.0, 2.0, 3.0])

    def test_fit_supports_multiple_series(self, estimator):
        data = make_data(
            series_ids=("series_1", "series_2"),
            times=range(10),
        )

        estimator.fit(data)

        assert set(estimator.series_ids_) == {"series_1", "series_2"}
        assert estimator.n_series_ == 2
        assert estimator.series_lengths_ == {
            "series_1": 10,
            "series_2": 10,
        }

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("->", "-->"),
            ("<-", "<--"),
            ("<>", "<->"),
        ],
    )
    def test_link_mark_aliases_are_normalized(self, estimator, alias, expected):
        assert estimator._convert_link_mark(alias) == expected

    @pytest.mark.parametrize(
        "link_mark",
        ["-->", "<--", "<->", "-?>", "o-o", "o?o", "<?>"],
    )
    def test_supported_link_marks_are_accepted(self, estimator, link_mark):
        assert estimator._convert_link_mark(link_mark) == link_mark

    def test_unsupported_link_mark_is_rejected(self, estimator):
        with pytest.raises(ValueError):
            estimator._convert_link_mark("invalid")

    def test_link_assumptions_are_normalized(self, estimator):
        estimator.n_variables_ = 2
        estimator.link_assumptions = {1: {(0, -1): "->"}}

        estimator._validate_link_assumptions()

        assert estimator.link_assumptions_ == {1: {(0, -1): "-->"}}
        assert estimator.link_assumptions == {1: {(0, -1): "->"}}

    @pytest.mark.parametrize(
        "link_assumptions",
        [
            {2: {(0, -1): "-->"}},
            {1: {(2, -1): "-->"}},
            {1: {(0, -2): "-->"}},
            {1: {(0, 1): "-->"}},
            {1: {(0, -1): "invalid"}},
            {"1": {(0, -1): "-->"}},
            {1: {"invalid": "-->"}},
        ],
        ids=[
            "invalid-child-index",
            "invalid-parent-index",
            "lag-below-minimum",
            "positive-lag",
            "invalid-link-mark",
            "non-integer-child-index",
            "invalid-parent-lag-key",
        ],
    )
    def test_invalid_link_assumptions_are_rejected(self, estimator, link_assumptions):
        estimator.n_variables_ = 2
        estimator.link_assumptions = link_assumptions

        with pytest.raises((TypeError, ValueError)):
            estimator._validate_link_assumptions()

    @pytest.mark.parametrize(
        ("tau_min", "tau_max"),
        [
            (-1, 1),
            (0, -1),
            (2, 1),
            (0.5, 1),
            (0, 1.5),
            (True, 1),
        ],
        ids=[
            "negative-minimum",
            "negative-maximum",
            "minimum-exceeds-maximum",
            "non-integer-minimum",
            "non-integer-maximum",
            "boolean-minimum",
        ],
    )
    def test_invalid_lag_parameters_are_rejected(self, estimator, tau_min, tau_max):
        estimator.tau_min = tau_min
        estimator.tau_max = tau_max

        with pytest.raises((TypeError, ValueError)):
            estimator._validate_parameters()

    def test_fit_requires_conditional_independence_test(self, estimator):
        estimator.cond_ind_test = None

        with pytest.raises((TypeError, ValueError)):
            estimator._validate_parameters()

    def test_reset_clears_fitted_state(self, estimator, data):
        estimator.fit(data)
        estimator._reset_fit_state()

        assert estimator.series_ids_ is None
        assert estimator.variable_names_ is None
        assert estimator.results_ is None
        assert estimator.graph_ is None
        assert estimator.p_matrix_ is None
        assert estimator.val_matrix_ is None
