import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

from pgmpy.ci_tests._tigramite_adapter import TigramiteAdapter


class BaseTimeSeriesCausalDiscovery(ABC):
    """
    Base class for time-series causal discovery algorithms.

    Input data follows sktime's MultiIndex DataFrame convention:

    Subclasses implement algorithm initialization, discovery,
    and result processing.
    """

    def __init__(
        self,
        tau_min=0,
        tau_max=1,
        cond_ind_test=None,
        verbosity=0,
    ):
        """Initialize the time-series causal discovery estimator."""

        if not isinstance(tau_min, int) or isinstance(tau_min, bool) or tau_min < 0:
            raise ValueError("tau_min must be a non negative integer")

        if not isinstance(tau_max, int) or isinstance(tau_max, bool) or tau_max < 0:
            raise ValueError("tau_max must be a non negative integer")

        if tau_min > tau_max:
            raise ValueError("Past can't be greater than future")

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.cond_ind_test = cond_ind_test
        self.verbosity = verbosity

        # Dataset metadata.
        self.series_ids_ = None
        self.variable_names_ = None
        self.n_variables_ = None
        self.n_observations_ = None
        self.n_series_ = None
        self.series_data_ = None
        self.series_lengths_ = None
        self.X_ = None

        # Discovery results.
        self.graph_ = None
        self.p_matrix_ = None
        self.val_matrix_ = None

    def _validate_time_series_data(self, X):
        """
        Validate the input time-series DataFrame.

        Checks the DataFrame structure, index integrity, variable names,
        numeric values, missing values, and time-point ordering.

        Returns
        -------
        pandas.DataFrame
            The validated input DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError(
                "Input DataFrame must use a MultiIndex with level 0 as the series identifier and level 1 as time."
            )

        if X.index.nlevels != 2:
            raise ValueError("Input MultiIndex must have exactly two levels.")

        if X.empty or X.shape[1] == 0:
            raise ValueError("Input DataFrame must contain observations and variables.")

        if X.columns.has_duplicates:
            raise ValueError("Input DataFrame must have unique variable names.")

        if not all(isinstance(name, str) for name in X.columns):
            raise TypeError("All variable names must be strings.")

        if any(X.index.get_level_values(level).hasnans for level in range(X.index.nlevels)):
            raise ValueError("Input MultiIndex must not contain missing identifiers.")

        if X.isna().to_numpy().any():
            raise ValueError("Input data contains missing values. Handle them before fitting.")

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError("All time-series variables must be numeric.")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Input data must contain only finite numeric values.")

        # Check each series independently.
        for series_id, group in X.groupby(level=0, sort=False):
            time_index = group.index.get_level_values(1)

            if time_index.has_duplicates:
                raise ValueError(f"Series {series_id!r} contains duplicate time points.")

            if not time_index.is_monotonic_increasing:
                warnings.warn(
                    f"Time points for series {series_id!r} are not ordered. They will be sorted before discovery.",
                    UserWarning,
                    stacklevel=2,
                )

        return X

    def _validate_series_lengths(self):
        """
        Ensure every independent time series has enough observations.

        Each series must contain more observations than tau_max.
        This prevents attempting lagged discovery on a series that is
        too short for the configured maximum lag.
        """
        if self.series_lengths_ is None:
            raise RuntimeError("Series lengths are unavailable. Prepare the data first.")

        too_short = [series_id for series_id, length in self.series_lengths_.items() if length <= self.tau_max]

        if too_short:
            raise ValueError(
                "Each time series must contain more observations than "
                f"tau_max. Series that are too short: {too_short!r}"
            )

    def _prepare_time_series_data(self, X):
        """
        Extract time-series metadata and prepare individual series.

        Assumes that X has already passed _validate_time_series_data().
        Each series is sorted by its time index before being stored.
        """
        self.series_ids_ = X.index.get_level_values(0).unique()
        self.variable_names_ = list(X.columns)

        self.n_variables_ = X.shape[1]
        self.n_observations_ = len(X)
        self.n_series_ = len(self.series_ids_)

        self.series_data_ = {series_id: X.xs(series_id, level=0).sort_index() for series_id in self.series_ids_}

        self.series_lengths_ = {series_id: len(series_data) for series_id, series_data in self.series_data_.items()}

        return self.series_data_

    def _wrap_ci_test_for_tigramite(self):
        """
        Adapt the configured conditional independence test for Tigramite.
        """
        if self.cond_ind_test is None:
            raise ValueError("cond_ind_test must be provided to perform causal discovery.")

        # Keep the configured test separate from the adapter.
        ci_test = deepcopy(self.cond_ind_test)

        self.tigramite_ci_test_ = TigramiteAdapter(
            tigramite_ci=ci_test,
            tau_max=self.tau_max,
            cut_off="2xtau_max",
            alpha_or_thres=None,
        )

        return self.tigramite_ci_test_

    @abstractmethod
    def _initialize_algorithm(self) -> None:
        """Initialize the concrete time-series discovery algorithm."""

    @abstractmethod
    def _run_discovery(self) -> dict[str, object]:
        """Run causal discovery and return raw results."""

    @abstractmethod
    def _process_discovery_results(self, results: dict[str, object]) -> None:
        """Normalize the results into the public estimator attributes."""

    def fit(self, X):
        """
        Fit the estimator to one or more time series.

        Parameters
        ----------
        X : pandas.DataFrame
            MultiIndex DataFrame containing time-series observations.

        Returns
        -------
        self
            The fitted estimator.
        """
        # 1. Validate the complete input.
        self._validate_time_series_data(X)

        # 2. Preserve the original input.
        self.X_ = X.copy()

        # 3. Extract individual series and metadata.
        self._prepare_time_series_data(X)

        # 4. Validate the lengths of individual series.
        self._validate_series_lengths()

        # 5. Adapt the conditional independence test.
        self._wrap_ci_test_for_tigramite()

        # 6. Initialize the algorithm.
        self._initialize_algorithm()

        # 7. Run causal discovery.
        results = self._run_discovery()

        # 8. Normalize and store the results.
        self._process_discovery_results(results=results)
        self.results_ = results

        return self

    def get_causal_links(self):
        """
        Return the discovered causal relationships.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if self.graph_ is None:
            raise RuntimeError("The estimator has not been fitted yet.")

        return self.graph_

    def get_parameters(self):
        """Return the estimator's configuration."""
        return {
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "cond_ind_test": self.cond_ind_test,
            "verbosity": self.verbosity,
        }
