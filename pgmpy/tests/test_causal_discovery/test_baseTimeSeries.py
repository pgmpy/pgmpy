import numpy as np
import pandas as pd

from pgmpy.causal_discovery._baseTimeSeries import BaseTimeSeriesCausalDiscovery
from pgmpy.ci_tests import Pearsonr


def _make_lagged_time_series(n_observations: int = 500) -> pd.DataFrame:
    """Generate a series where X at t-1 causes Y at t."""
    rng = np.random.default_rng(42)
    x = np.zeros(n_observations)
    y = np.zeros(n_observations)

    for t in range(1, n_observations):
        x[t] = 0.7 * x[t - 1] + rng.normal()
        y[t] = 0.8 * x[t - 1] + 0.3 * y[t - 1] + rng.normal(scale=0.5)

    index = pd.MultiIndex.from_product(
        [["series_1"], range(n_observations)],
        names=["series_id", "time"],
    )
    return pd.DataFrame({"X": x, "Y": y}, index=index)


class _CoordinatorTestEstimator(BaseTimeSeriesCausalDiscovery):
    """Provide predictable algorithm hooks to test the base fit coordinator."""

    def __init__(self, data: pd.DataFrame):
        super().__init__(cond_ind_test=Pearsonr(data=data), tau_max=1)
        self.expected_results = {
            "graph": "synthetic-link",
            "p_matrix": np.zeros((2, 2, 2)),
            "val_matrix": np.zeros((2, 2, 2)),
        }
        self.hook_calls = []

    def _initialize_algorithm(self) -> None:
        self.hook_calls.append("initialize")

    def _run_discovery(self) -> dict[str, object]:
        self.hook_calls.append("run")
        return self.expected_results

    def _process_discovery_results(self, results: dict[str, object]) -> None:
        self.hook_calls.append("process")
        self.graph_ = results["graph"]
        self.p_matrix_ = results["p_matrix"]
        self.val_matrix_ = results["val_matrix"]


def test_base_time_series_fit_coordinates_discovery():
    data = _make_lagged_time_series()
    estimator = _CoordinatorTestEstimator(data)

    assert np.corrcoef(data["X"].to_numpy()[:-1], data["Y"].to_numpy()[1:])[0, 1] > 0.5

    result = estimator.fit(data)

    assert result is estimator
    assert estimator.hook_calls == ["initialize", "run", "process"]
    assert estimator.series_ids_ is not None
    assert estimator.series_ids_.tolist() == ["series_1"]
    assert estimator.variable_names_ == ["X", "Y"]
    assert estimator.n_variables_ == 2
    assert estimator.n_observations_ == len(data)
    assert estimator.n_series_ == 1
    assert estimator.series_lengths_ == {"series_1": len(data)}
    assert estimator.series_data_ is not None
    np.testing.assert_array_equal(estimator.series_data_["series_1"].to_numpy(), data.to_numpy())
    assert estimator.ci_test is not estimator.cond_ind_test
    assert estimator.tigramite_ci_test_.cond_ind_test is estimator.ci_test
    assert estimator.results_ is estimator.expected_results
    assert estimator.get_causal_links() == "synthetic-link"
