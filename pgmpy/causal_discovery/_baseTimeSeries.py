import pandas as pd


class BaseTimeSeriesCausalDiscovery:
    """
    Base class for time-series causal discovery algorithms.
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
        pass

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
        To Be Implemented.
        """

    def _initialize_algorithm(self):
        """
        Initialize the underlying time-series causal discovery algorithm.

        Configure the selected implementation with the prepared data,
        temporal constraints, and wrapped conditional independence test.

        Delegate algorithm-specific configuration to the concrete
        estimator rather than assuming every implementation exposes
        identical parameters.
        """

    def _run_discovery(self):
        pass

    def _process_discovery_results(self, results):
        """
        Convert the underlying algorithm's outputs into the result
        conventions exposed by pgmpy's time-series causal discovery API.
        """

    def fit(self, X):
        """
        Fit the time-series causal discovery estimator.

        Execute the common lifecycle:

        1. Validate the input representation.
        2. Validate temporal and statistical configuration.
        3. Prepare the time-series data and metadata.
        4. Construct the conditional independence test adapter.
        5. Initialize and execute the underlying discovery algorithm.
        6. Process and store the resulting causal relationships.

        Store fitted state using trailing-underscore attribute names,
        following common Python estimator conventions.

        Return the fitted estimator to support method chaining.
        """

    def get_causal_links(self):
        """
        Return the discovered causal relationships.
        """

    def get_parameters(self):
        """
        Return the configuration used for time-series causal discovery.
        """
