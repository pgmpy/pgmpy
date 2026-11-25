import numpy as np

from pgmpy.causal_discovery._baseTimeSeries import BaseTimeSeriesCausalDiscovery
from pgmpy.ci_tests._tigramite_adapter import TigramiteAdapter


class _BasePCMCI(BaseTimeSeriesCausalDiscovery):
    """
    Base class for PCMCI-based time-series causal discovery algorithms.

    This class integrates Tigramite's PCMCI implementation with pgmpy's
    time-series causal discovery interface.

    Multiple time-series realizations are passed to Tigramite as separate
    datasets. The native Tigramite conditional independence test is used
    internally by PCMCI and is also exposed through pgmpy's CI-test
    interface using ``TigramiteAdapter``.

    Parameters
    ----------
    cond_ind_test : object, optional
        Tigramite conditional independence test instance. If None,
        Tigramite's ParCorr test is used.

    tau_min : int, default=0
        Minimum time lag considered during causal discovery.

    tau_max : int, default=1
        Maximum time lag considered during causal discovery.

    pc_alpha : float, list, or None, default=0.05
        Significance level used during the PC condition selection phase.
        Can also be a list of significance levels or None, depending on
        the Tigramite version.

    fdr_method : str, default="fdr_bh"
        Multiple-testing correction method used by PCMCI.

    link_assumptions : dict, optional
        Prior assumptions about the existence and orientation of links.

    **kwargs
        Additional keyword arguments forwarded to Tigramite's
        ``run_pcmci`` method.

    Attributes
    ----------
    tigramite_data_ : tigramite.data_processing.DataFrame
        Tigramite representation of the input time-series data.

    pcmci_ : tigramite.pcmci.PCMCI
        Initialized Tigramite PCMCI instance.

    ci_test_ : TigramiteAdapter
        Pgmpy-compatible wrapper around the native Tigramite CI test.

    tigramite_ci_test_ : object
        Native Tigramite conditional independence test.

    graph_ : numpy.ndarray
        Discovered causal graph.

    p_matrix_ : numpy.ndarray
        Matrix of p-values for discovered links.

    val_matrix_ : numpy.ndarray
        Matrix of test statistics for discovered links.

    Examples
    --------
    >>> estimator = _BasePCMCI(tau_max=2)
    >>> estimator.fit(data)
    """

    def __init__(
        self,
        cond_ind_test=None,
        tau_min=0,
        tau_max=1,
        pc_alpha=0.05,
        fdr_method="fdr_bh",
        link_assumptions=None,
        **kwargs,
    ):
        super().__init__(
            tau_min=tau_min,
            tau_max=tau_max,
            cond_ind_test=cond_ind_test,
            link_assumptions=link_assumptions,
        )

        self.pc_alpha = pc_alpha
        self.fdr_method = fdr_method
        self.kwargs = kwargs

    def _reset_fit_state(self):
        """Reset algorithm-specific and inherited fitted attributes.

        Examples
        --------
        >>> estimator._reset_fit_state()
        >>> estimator.pcmci_ is None
        True
        """
        super()._reset_fit_state()

        self.tigramite_data_ = None
        self.pcmci_ = None

    def _validate_parameters(self):
        """Validate PCMCI-specific parameters.

        The common time-series parameters are validated by the parent
        class. A missing conditional independence test is allowed because
        the default Tigramite test is initialized during fitting.

        Examples
        --------
        >>> estimator = _BasePCMCI()
        >>> estimator._validate_parameters()
        """
        super()._validate_parameters()

        if self.pc_alpha is not None:
            if isinstance(self.pc_alpha, (list, tuple)):
                if not self.pc_alpha:
                    raise ValueError("pc_alpha cannot be an empty list.")

                for alpha in self.pc_alpha:
                    if not isinstance(alpha, (int, float, np.number)):
                        raise TypeError("Every pc_alpha value must be numeric.")

                    if isinstance(alpha, (bool, np.bool_)) or not 0 < alpha < 1:
                        raise ValueError("Every pc_alpha value must be between 0 and 1.")
            else:
                if not isinstance(self.pc_alpha, (int, float, np.number)):
                    raise TypeError("pc_alpha must be numeric, a list, or None.")

                if isinstance(self.pc_alpha, (bool, np.bool_)) or not 0 < self.pc_alpha < 1:
                    raise ValueError("pc_alpha must be between 0 and 1.")

    def _initialize_algorithm(self):
        """Initialize Tigramite's data and PCMCI objects.

        Each time-series realization is passed independently to Tigramite.
        The input data is not concatenated across realizations.

        If no CI test was provided, Tigramite's ParCorr test is used.

        The native Tigramite test is passed directly to PCMCI. A
        ``TigramiteAdapter`` is additionally created to expose the test
        through pgmpy's CI-test interface.

        Examples
        --------
        >>> estimator._initialize_algorithm()
        """
        from tigramite import data_processing as dp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        # Default native Tigramite CI test.
        if self.cond_ind_test is None:
            self.tigramite_ci_test_ = ParCorr()
        else:
            self.tigramite_ci_test_ = self.cond_ind_test

        # Expose the native Tigramite test through pgmpy's CI interface.
        self.ci_test_ = TigramiteAdapter(
            tigramite_ci=self.tigramite_ci_test_,
            variable_names=self.variable_names_,
            tau_max=self.tau_max,
        )

        # Preserve individual realizations as separate datasets.
        data = {index: series.to_numpy(dtype=float) for index, series in enumerate(self.series_data_.values())}

        self.tigramite_data_ = dp.DataFrame(
            data=data,
            var_names=self.variable_names_,
        )

        self.pcmci_ = PCMCI(
            dataframe=self.tigramite_data_,
            cond_ind_test=self.tigramite_ci_test_,
        )

    def _run_discovery(self, **fit_params):
        """Execute the PCMCI causal discovery procedure.

        Parameters
        ----------
        **fit_params : dict
            Additional arguments passed through the inherited fit
            interface and forwarded to Tigramite's ``run_pcmci``.

        Returns
        -------
        dict
            Raw result dictionary returned by Tigramite.

        Examples
        --------
        >>> results = estimator._run_discovery()
        """
        run_kwargs = {
            **self.kwargs,
            **fit_params,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "fdr_method": self.fdr_method,
            "link_assumptions": self.link_assumptions_,
        }

        return self.pcmci_.run_pcmci(**run_kwargs)

    def _process_discovery_results(self, results):
        """Validate and store the results returned by Tigramite.

        Parameters
        ----------
        results : dict
            Raw output returned by ``PCMCI.run_pcmci``.

        Raises
        ------
        TypeError
            If the result is not a dictionary.
        ValueError
            If required result fields are missing or their shapes
            are inconsistent with the input data.

        Examples
        --------
        >>> estimator._process_discovery_results(results)
        """
        if not isinstance(results, dict):
            raise TypeError("PCMCI results must be returned as a dictionary.")

        required_keys = {"graph", "p_matrix", "val_matrix"}
        missing_keys = required_keys.difference(results)

        if missing_keys:
            raise ValueError(f"PCMCI results are missing required keys: {sorted(missing_keys)}.")

        graph = np.asarray(results["graph"])
        p_matrix = np.asarray(results["p_matrix"])
        val_matrix = np.asarray(results["val_matrix"])

        expected_shape = (
            self.n_variables_,
            self.n_variables_,
            self.tau_max + 1,
        )

        for name, value in (
            ("graph", graph),
            ("p_matrix", p_matrix),
            ("val_matrix", val_matrix),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"Invalid shape for {name}. Expected {expected_shape}, got {value.shape}.")

        self.graph_ = graph
        self.p_matrix_ = p_matrix
        self.val_matrix_ = val_matrix
