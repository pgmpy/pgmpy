import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

from pgmpy.causal_discovery._tigramite_adapter import PgmpyTigramiteCITest


class BaseTimeSeriesCausalDiscovery(ABC):
    """
    Base class for time-series causal discovery algorithms.

    Input data follows sktime's multi-index dataframe convention.
    Level 0 identifies an independent series, level 1 identifies the grid
    of rows and columns which are variables.

    Subclasses implement the algoritm-specific initialization of time series
    causal discovery and result processing methods.
    """

    def __init__(
        self,
        tau_min=0,
        tau_max=1,
        cond_ind_test=None,
        verbosity=0,
    ):
        """
        Initialize the base time-series causal discovery estimator.

        Parameters
        ----------
        tau_min : int
            Minimum temporal lag considered during causal discovery.

        tau_max : int
            Maximum temporal lag considered during causal discovery.

        cond_ind_test : object
            Conditional independence test used to evaluate candidate
            causal relationships. The test is expected to follow pgmpy's
            conditional independence testing interface.

        verbosity : int
            Controls the amount of diagnostic information emitted during
            the discovery process.
        """
        # validate temporal lag configuration
        if not isinstance(tau_min, int) or isinstance(tau_min, bool) or tau_min < 0:
            raise ValueError("tau_min must be a non negative integer")

        if not isinstance(tau_max, int) or isinstance(tau_max, bool) or tau_max < 0:
            raise ValueError("tau_max must be a non negative integer")

        if tau_min > tau_max:
            raise ValueError("Past can't be greater than future")

        self.tau_min = tau_min
        self.tau_max = tau_max
        # conditional independence test
        self.cond_ind_test = cond_ind_test
        self.verbosity = verbosity

        # initialize dataset metadata, populated when estimator receives data
        self.series_ids_ = None
        self.variable_names_ = None
        self.n_variables_ = None
        self.n_observations_ = None
        self.n_series_ = None
        self.series_data_ = None
        self.series_lengths_ = None
        self.X_ = None  # for creating a dataset copy

        # values populated after fitting
        self.graph_ = None
        self.p_matrix_ = None
        self.val_matrix_ = None

    def _validate_time_series_data(self, X):
        """Validate the sktime multi-index DataFrame."""
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

        if X.index.hasnans:
            raise ValueError("Input MultiIndex must not contain missing identifiers.")

        if not all(isinstance(name, str) for name in X.columns):
            raise TypeError("All variable names must be strings.")

        if X.columns.has_duplicates:
            raise ValueError("Variable names must be unique.")

        if X.isna().to_numpy().any():
            raise ValueError("Input data contains missing values. Handle them before fitting.")

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError("All time-series variables must be numeric.")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Input data must contain only finite numeric values.")

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

    def _prepare_time_series_data(self, X):
        """
        Prepare time series multi-index DataFrame for discovery.

        The input dataframe is expected to use a multi-index with:
        - level 0: instance or series identifier
        - level 1: time-point identifier

        Columns represent variables whose causal relationships are to be discovered.
        """

        # ensure input is pandas dataframe
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # ensure df uses a multi-index
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("Input DataFrame must have a MultiIndex.")

        # ensure data not empty
        if X.empty or X.shape[1] == 0:
            raise ValueError("Input DataFrame must contain at least one variable.")

        # extract identifiers and names
        self.series_ids_ = X.index.get_level_values(0).unique()
        self.variable_names_ = list(X.columns)

        # Number of variables and total observations
        self.n_variables_ = X.shape[1]
        self.n_observations_ = len(X)

        # number of independent series
        self.n_series_ = len(self.series_ids_)

        self.series_data_ = {series_id: X.xs(series_id, level=0).sort_index() for series_id in self.series_ids_}

        self.series_lengths_ = {series_id: len(series_data) for series_id, series_data in self.series_data_.items()}

        # return prepared representation
        return self.series_data_

    def _wrap_ci_test_for_tigramite(self):
        """
        Adapt pgmpy's CI-test callable to Tigramite's run_test interface.

        The adapter is created by a subclass once its data representation
        and algorithm-specific requirements are known.

        Tigramite's standard CI-test protocol expects a callable with
        set_dataframe(dataframe) and run_test(X, Y, Z, tau_max, ...).
        """
        if self.cond_ind_test is None:
            raise ValueError("cond_ind_test must be provided to perform causal discovery.")

        self.tigramite_ci_test_ = PgmpyTigramiteCITest(
            cond_ind_test=self.tigramite_ci_test_, variable_names=self.variable_names_
        )
        return self.tigramite_ci_test_

    @abstractmethod
    def _initialize_algorithm(self):
        """Initialize the concrete time-series discovery methods."""

    @abstractmethod
    def _run_discovery(self):
        """Run the concrete discovery algorithm and return raw results."""

    @abstractmethod
    def _process_discovery_results(self, results):
        """Normalize the results into the public pgmpy result attributes."""

    def fit(self, X):
        """
        Fit the estimator to one or more time series.

        Algorithm-specific methods receive the prepared series and own
        their Tigramite configuration and result processing.
        """
        self._prepare_time_series_data(X)

        self.ci_test = deepcopy(self.cond_ind_test)
        self.tigramite_ci_test_ = self._wrap_ci_test_for_tigramite()

        self._initialize_algorithm()
        results = self._run_discovery()
        self._process_discovery_results(results=results)
        self.results_ = results

        return self

    def get_causal_links(self):
        """
        Return the discovered causal relationships.
        """
        if self.graph_ is None:
            raise RuntimeError("The estimator has not been fitted yet")
        return self.graph_

    def get_parameters(self):
        """
        Return the configuration used for time-series causal discovery.
        """
        return {
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "cond_ind_test": self.cond_ind_test,
            "verbosity": self.verbosity,
        }
