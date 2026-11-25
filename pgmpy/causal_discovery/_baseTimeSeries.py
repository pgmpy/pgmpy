import warnings
from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils.validation import check_array

from pgmpy.causal_discovery._base import BaseCausalDiscovery

TIGRAMITE_LINK_MARKS = {
    "-->",
    "<--",
    "<->",
    "-?>",
    "o-o",
    "o?o",
    "<?>",
}

LINK_MARK_ALIASES = {
    "->": "-->",
    "<-": "<--",
    "<>": "<->",
}


class BaseTimeSeriesCausalDiscovery(BaseCausalDiscovery, ABC):
    """
    Base class for time-series causal discovery estimators.

    Extends pgmpy's BaseCausalDiscovery with support for multiple
    time series represented by a two-level MultiIndex DataFrame.

    The first index level identifies the time series, and the second
    identifies the time point. Each column represents one observed variable.

    Examples
    --------
    A valid input has one row per series and time point::

        index = pd.MultiIndex.from_tuples(
            [("subject_1", 0), ("subject_1", 1),
             ("subject_2", 0), ("subject_2", 1)],
            names=["series_id", "time"],
        )
        data = pd.DataFrame(
            {"temperature": [20.1, 20.4, 19.8, 20.0]},
            index=index,
        )

    A concrete subclass implements the algorithm hooks and is fitted
    through the inherited estimator interface::

        estimator = ConcreteTimeSeriesDiscovery(...)
        estimator.fit(data)

    Parameters
    ----------
    tau_min : int, default=0
        Minimum time lag considered during causal discovery.

    tau_max : int, default=1
        Maximum time lag considered during causal discovery.

    cond_ind_test : object
        Conditional independence test used by the discovery algorithm.

    link_assumptions : dict, optional
        Prior assumptions about the existence and orientation of
        links between variables at different time lags.

    Notes
    -----
    This class inherits the fitting interface and general data
    validation lifecycle from pgmpy's BaseCausalDiscovery.

    Subclasses are responsible for implementing algorithm-specific
    initialization, execution, and result processing.
    """

    def __init__(
        self,
        tau_min=0,
        tau_max=1,
        cond_ind_test=None,
        link_assumptions=None,
    ):
        """Initialize the shared time-series causal discovery configuration.

        Parameters
        ----------
        tau_min : int, default=0
            Smallest lag considered by the discovery algorithm.

        tau_max : int, default=1
            Largest lag considered by the discovery algorithm.

        cond_ind_test : object, optional
            Conditional independence test configuration used by the
            implementation. The current base-class parameter validation
            requires this value to be non-None.

        link_assumptions : dict, optional
            Prior constraints on links, keyed by child variable index.
            Each child maps ``(parent_variable_index, lag)`` pairs to
            link marks.

        Examples
        --------
        A concrete subclass can be configured through the inherited estimator API::

            estimator = ConcreteTimeSeriesDiscovery(
                tau_min=0,
                tau_max=2,
                cond_ind_test=my_ci_test,
            )
            estimator.get_params()["tau_max"]  # 2
        """
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.cond_ind_test = cond_ind_test
        self.link_assumptions = link_assumptions

    def _reset_fit_state(self):
        """Reset attributes populated during fitting.

        Clears time-series metadata and algorithm outputs before a new fit
        attempt. This prevents results from a previous fit being mistaken
        for results from a subsequent fit.

        Examples
        --------
        >>> from types import SimpleNamespace
        >>> state = SimpleNamespace()
        >>> BaseTimeSeriesCausalDiscovery._reset_fit_state(state)
        >>> state.series_ids_ is None
        True
        >>> state.graph_ is None
        True
        """

        self.series_ids_ = None
        self.variable_names_ = None

        self.n_variables_ = None
        self.n_observations_ = None
        self.n_series_ = None

        self.series_data_ = None
        self.series_lengths_ = None
        self.X_ = None

        self.ci_test_ = None
        self.tigramite_ci_test_ = None

        self.link_assumptions_ = None
        self.results_ = None

        self.graph_ = None
        self.p_matrix_ = None
        self.val_matrix_ = None

    def _validate_parameters(self):
        """Validate the configured time-lag parameters.

        The lag bounds must be non-negative integers (booleans are rejected),
        and ``tau_min`` must not exceed ``tau_max``. This implementation also
        requires ``cond_ind_test`` to be configured.

        Raises
        ------
        ValueError
            If a lag parameter is invalid, the lag range is reversed, or
            ``cond_ind_test`` is ``None``.

        Examples
        --------
        >>> from types import SimpleNamespace
        >>> params = SimpleNamespace(
        ...     tau_min=0,
        ...     tau_max=2,
        ...     cond_ind_test=object(),
        ... )
        >>> BaseTimeSeriesCausalDiscovery._validate_parameters(params)
        """

        for name in ("tau_min", "tau_max"):
            value = getattr(self, name)

            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer. Got {value!r}.")

        if self.tau_min > self.tau_max:
            raise ValueError("tau_min cannot be greater than tau_max.")

    def _check_fit_data(self, X):
        """Validate input and preserve its time-series index structure.

        Requires a pandas DataFrame with a two-level MultiIndex. The first
        level identifies a series and the second identifies time. Variable
        names must be unique strings. General validation and preprocessing
        are delegated to ``BaseCausalDiscovery``; this method additionally
        checks that preprocessing preserves the DataFrame and original index.

        Parameters
        ----------
        X : pandas.DataFrame
            Observations indexed by ``(series_id, time)``.

        Returns
        -------
        pandas.DataFrame
            The validated input.

        Raises
        ------
        TypeError
            If the input is not a DataFrame or its column names are not
            strings, or preprocessing changes the container type.
        ValueError
            If the index is not a two-level MultiIndex, columns are
            duplicated, or preprocessing changes the index.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [("series_a", 0), ("series_a", 1)],
        ...     names=["series", "time"],
        ... )
        >>> frame = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
        >>> frame.index.nlevels
        2
        """

        # Reset state before parent validation. This also ensures
        # failed refits do not retain previous fitted attributes.
        self._reset_fit_state()

        self._validate_parameters()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
            raise ValueError("Input DataFrame must have a two-level MultiIndex with series identifier and time.")

        if X.columns.has_duplicates:
            raise ValueError("Variable names must be unique.")

        if not X.columns.map(lambda name: isinstance(name, str)).all():
            raise TypeError("All variable names must be strings.")

        for column in X.columns:
            if not is_numeric_dtype(X[column].dtype):
                raise TypeError(f"Column '{column}' must contain numeric values.")

        original_index = X.index

        # Reuse pgmpy's complete validation and preprocessing logic.
        X = super()._check_fit_data(X)

        # Time-series algorithms require the original MultiIndex.
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "pgmpy's data preprocessing must preserve the pandas DataFrame representation for time-series data."
            )

        if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
            raise ValueError("Data preprocessing must preserve the two-level time-series MultiIndex.")

        if not X.index.equals(original_index):
            raise ValueError("Data preprocessing must preserve the original series identifiers and time index.")

        return X

    def _prepare_data(self, X):
        """Prepare individual time series and populate fitted metadata.

        Checks that index values are present and observations are numeric,
        finite, and non-complex. Each series is extracted independently;
        duplicate time points are rejected and unsorted time indexes are
        sorted with a warning. Every series must contain more observations
        than ``tau_max``. The method then reconstructs the combined DataFrame
        and records series and variable metadata.

        Parameters
        ----------
        X : pandas.DataFrame
            Validated input with a two-level MultiIndex.

        Attributes Created
        ------------------
        series_ids_ : pandas.Index
            Unique series identifiers.

        series_data_ : dict
            Mapping from series identifier to its sorted DataFrame.

        series_lengths_ : dict
            Number of observations in each series.

        X_ : pandas.DataFrame
            Reconstructed DataFrame containing the sorted series.

        variable_names_ : list
            Names of the observed variables.

        n_variables_ : int
            Number of variables.

        n_observations_ : int
            Total number of observations.

        n_series_ : int
            Number of series.

        Raises
        ------
        ValueError
            If the index contains missing values, a series has duplicate
            time points, or a series has at most ``tau_max`` observations.

        TypeError
            If observations are complex or cannot be converted to numeric
            values.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [("A", 0), ("A", 1), ("A", 2),
        ...      ("B", 0), ("B", 1), ("B", 2)],
        ...     names=["series", "time"],
        ... )
        >>> frame = pd.DataFrame({"x": [1., 2., 3., 4., 5., 6.]}, index=idx)
        >>> frame.index.get_level_values("series").nunique()
        2
        """

        if X.index.to_frame(index=False).isna().to_numpy().any():
            raise ValueError("Input index must not contain missing values.")

        # Tigramite requires numerical observations.
        if np.iscomplexobj(X.to_numpy()):
            raise TypeError("Complex-valued data is not supported.")

        check_array(
            X,
            dtype="numeric",
            ensure_2d=True,
            ensure_all_finite=True,
            ensure_min_samples=1,
            ensure_min_features=1,
        )

        self.series_ids_ = X.index.get_level_values(0).unique()

        self.series_data_ = {}

        for series_id in self.series_ids_:
            data = X.xs(series_id, level=0)
            time_index = data.index

            if time_index.has_duplicates:
                raise ValueError(f"Series {series_id!r} contains duplicate time points.")

            if not time_index.is_monotonic_increasing:
                warnings.warn(
                    f"Series {series_id!r} is not sorted by time. It will be sorted before discovery.",
                    UserWarning,
                    stacklevel=2,
                )

            self.series_data_[series_id] = data.sort_index()

        self.series_lengths_ = {series_id: len(data) for series_id, data in self.series_data_.items()}

        too_short = [series_id for series_id, length in self.series_lengths_.items() if length <= self.tau_max]

        if too_short:
            raise ValueError(
                f"Each series must contain more observations than tau_max. Series that are too short: {too_short!r}"
            )

        # Reconstruct the complete time-series DataFrame.
        self.X_ = pd.concat(
            self.series_data_,
            names=X.index.names,
        )

        self.variable_names_ = list(self.X_.columns)
        self.n_variables_ = self.X_.shape[1]
        self.n_observations_ = len(self.X_)
        self.n_series_ = len(self.series_ids_)

    def _convert_link_mark(self, mark):
        """Convert a link mark or alias to canonical Tigramite notation.

        Supported aliases are normalized using ``LINK_MARK_ALIASES``.
        Values not listed in ``TIGRAMITE_LINK_MARKS`` are rejected.

        Parameters
        ----------
        mark : str
            A supported link mark or alias.

        Returns
        -------
        str
            The canonical Tigramite link mark.

        Raises
        ------
        TypeError
            If ``mark`` is not a string.
        ValueError
            If the mark is not supported.

        Examples
        --------
        >>> BaseTimeSeriesCausalDiscovery._convert_link_mark(None, "->")
        '-->'
        >>> BaseTimeSeriesCausalDiscovery._convert_link_mark(None, "<>")
        '<->'
        """

        if not isinstance(mark, str):
            raise TypeError("Link marks must be strings.")

        mark = LINK_MARK_ALIASES.get(mark, mark)

        if mark not in TIGRAMITE_LINK_MARKS:
            raise ValueError(f"Unsupported link mark: {mark!r}")

        return mark

    def _validate_link_assumptions(self):
        """Validate and normalize the supplied link assumptions.

        The outer dictionary maps a child variable index to a dictionary of
        parent specifications. Each parent is represented by
        ``(parent_variable_index, lag)`` and maps to a link mark. Variable
        indices must refer to columns in the fitted data, and lags must be
        within ``[-tau_max, -tau_min]``. Normalized assumptions are stored
        in ``link_assumptions_``. If no assumptions were supplied, that
        attribute is set to ``None``.

        Examples
        --------
        For data with columns ``["X", "Y"]`` and ``tau_max=2``::

            link_assumptions = {
                1: {
                    (0, -1): "-->",
                    (1, -2): "o-o",
                }
            }

        The alias ``"->"`` is also accepted and normalized to ``"-->"``.
        """

        if self.link_assumptions is None:
            self.link_assumptions_ = None
            return

        if not isinstance(self.link_assumptions, dict):
            raise TypeError("link_assumptions must be a dictionary.")

        assumptions = {}

        for child, parents in self.link_assumptions.items():
            if not isinstance(child, Integral) or isinstance(child, (bool, np.bool_)):
                raise TypeError("Child indices must be integers.")

            child = int(child)

            if not 0 <= child < self.n_variables_:
                raise ValueError(f"Invalid child index: {child}.")

            if not isinstance(parents, dict):
                raise TypeError(f"Link assumptions for child {child} must be a dictionary.")

            assumptions[child] = {}

            for parent, mark in parents.items():
                if not isinstance(parent, tuple) or len(parent) != 2:
                    raise ValueError("Each parent must be represented as (variable, lag).")

                variable, lag = parent

                if not isinstance(variable, Integral) or isinstance(variable, (bool, np.bool_)):
                    raise TypeError("Parent variable indices must be integers.")

                if not isinstance(lag, Integral) or isinstance(lag, (bool, np.bool_)):
                    raise TypeError("Parent lags must be integers.")

                variable = int(variable)
                lag = int(lag)

                if not 0 <= variable < self.n_variables_:
                    raise ValueError(f"Invalid parent variable index: {variable}.")

                if not -self.tau_max <= lag <= -self.tau_min:
                    raise ValueError(f"Invalid lag {lag}. Expected a lag between {-self.tau_max} and {-self.tau_min}.")

                assumptions[child][(variable, lag)] = self._convert_link_mark(mark)

        self.link_assumptions_ = assumptions

    @abstractmethod
    def _initialize_algorithm(self):
        """Initialize the concrete causal discovery backend.

        Subclasses should create and configure their algorithm-specific
        objects here. Shared data preparation and link-assumption validation
        have already completed when this hook is called from ``_fit``.

        Examples
        --------
        A concrete implementation might initialize a backend as follows::

            def _initialize_algorithm(self):
                self.algorithm_ = SomeTimeSeriesAlgorithm(
                    tau_min=self.tau_min,
                    tau_max=self.tau_max,
                    data=self.series_data_,
                )
        """

    @abstractmethod
    def _run_discovery(self, **fit_params):
        """Execute causal discovery and return the raw backend results.

        Parameters
        ----------
        **fit_params
            Additional fitting options forwarded from the public ``fit``
            method. Their interpretation is defined by the concrete
            algorithm.

        Returns
        -------
        object
            Raw results from the underlying discovery implementation.
            These are passed to ``_process_discovery_results``.

        Examples
        --------
        A concrete implementation may forward options to its backend::

            def _run_discovery(self, **fit_params):
                return self.algorithm_.run(**fit_params)
        """

    @abstractmethod
    def _process_discovery_results(self, results):
        """Convert raw backend results into estimator attributes.

        Concrete implementations should translate backend-specific output
        into the result representation exposed by the estimator. The
        implementation is expected to populate ``graph_``, ``p_matrix_``,
        and ``val_matrix_`` when those quantities are provided by the
        underlying algorithm.

        Parameters
        ----------
        results : object
            Raw result object returned by ``_run_discovery``.

        Examples
        --------
        A concrete implementation might extract result fields as follows::

            def _process_discovery_results(self, results):
                self.graph_ = results.graph
                self.p_matrix_ = results.p_matrix
                self.val_matrix_ = results.val_matrix
        """

    def _fit(self, X, **fit_params):
        """Run the shared time-series fitting workflow.

        This method is called by the inherited public ``fit`` method after
        pgmpy's general input validation. It prepares the time-series data,
        validates link assumptions, delegates backend initialization and
        execution to subclass hooks, and processes the raw results.

        Parameters
        ----------
        X : pandas.DataFrame
            Validated observations with a two-level MultiIndex containing
            series identifiers and time points.
        **fit_params
            Additional options forwarded to ``_run_discovery``.

        Returns
        -------
        self : BaseTimeSeriesCausalDiscovery
            The fitted estimator instance.

        Examples
        --------
        A concrete estimator is normally used through the inherited API::

            estimator = ConcreteTimeSeriesDiscovery(...)
            estimator.fit(time_series_dataframe)

        The concrete class implements the algorithm initialization,
        execution, and result-processing hooks.
        """

        self._prepare_data(X)
        self._validate_link_assumptions()
        self._initialize_algorithm()
        self.results_ = self._run_discovery(**fit_params)
        self._process_discovery_results(self.results_)

        return self
