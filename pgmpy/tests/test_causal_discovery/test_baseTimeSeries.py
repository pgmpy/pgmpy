from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery._baseTimeSeries import BaseTimeSeriesCausalDiscovery


class TestBaseTimeSeriesCausalDiscovery:
    @pytest.fixture
    def data(self):
        """Generate a time series where X[t-1] causes Y[t]."""
        rng = np.random.default_rng(42)
        n_observations = 500

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

    @pytest.fixture
    def estimator(self):
        """Create an estimator with predictable discovery hooks."""

        class CoordinatorTestEstimator(BaseTimeSeriesCausalDiscovery):
            def __init__(self):
                super().__init__(
                    cond_ind_test=SimpleNamespace(set_dataframe=lambda: None),
                    tau_max=1,
                )

                self.expected_results = {
                    "graph": "synthetic-link",
                    "p_matrix": np.zeros((2, 2, 2)),
                    "val_matrix": np.zeros((2, 2, 2)),
                }
                self.hook_calls = []

            def _initialize_algorithm(self):
                self.hook_calls.append("initialize")

            def _run_discovery(self):
                self.hook_calls.append("run")
                return self.expected_results

            def _process_discovery_results(self, results):
                self.hook_calls.append("process")
                self.graph_ = results["graph"]
                self.p_matrix_ = results["p_matrix"]
                self.val_matrix_ = results["val_matrix"]

        return CoordinatorTestEstimator()

    def test_fit_coordinates_discovery(self, data, estimator):
        """Test the fit coordinator and stored discovery results."""

        correlation = np.corrcoef(
            data["X"].to_numpy()[:-1],
            data["Y"].to_numpy()[1:],
        )[0, 1]

        assert correlation > 0.5

        result = estimator.fit(data)

        assert result is estimator
        assert estimator.hook_calls == [
            "initialize",
            "run",
            "process",
        ]

        # Check time-series metadata.
        assert estimator.series_ids_.tolist() == ["series_1"]
        assert estimator.variable_names_ == ["X", "Y"]
        assert estimator.n_variables_ == 2
        assert estimator.n_observations_ == len(data)
        assert estimator.n_series_ == 1
        assert estimator.series_lengths_ == {"series_1": len(data)}

        # Check stored series data.
        np.testing.assert_array_equal(
            estimator.series_data_["series_1"].to_numpy(),
            data.to_numpy(),
        )

        # Check conditional independence test configuration.
        assert estimator.ci_test is not estimator.cond_ind_test
        assert estimator.tigramite_ci_test_ is estimator.ci_test

        # Check discovery results.
        assert estimator.results_ is estimator.expected_results
        assert estimator.get_causal_links() == "synthetic-link"
