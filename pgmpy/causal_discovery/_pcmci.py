from abc import abstractmethod

from pgmpy.causal_discovery._baseTimeSeries import BaseTimeSeriesCausalDiscovery


class _BasePCMCI(BaseTimeSeriesCausalDiscovery):
    """Base class wrapping Tigramite's PCMCI implementation."""

    def __init__(
        self,
        cond_ind_test,
        tau_min=0,
        tau_max=1,
        pc_alpha=0.5,
        fdr_method="fdr_bh",
        verbosity=0,
        link_assumptions=None,
        **kwargs,
    ):
        """Initialize PCMCI-specific parameters."""
        super().__init__(
            tau_min=tau_min,
            tau_max=tau_max,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity,
        )
        self.pc_alpha = pc_alpha
        self.fdr_method = fdr_method
        self.link_assumptions = link_assumptions
        self.kwargs = kwargs

    def _prepare_data(self, X):
        """Convert the sktime multi-index DataFrame into Tigramite's input format."""
        pass

    def _wrap_ci_test_for_tigramite(self):
        """Adapt pgmpy's CI test to Tigramite's expected interface."""
        pass

    def _set_link_assumptions(self, link_assumptions=None):
        """Validate and convert temporal link assumptions for Tigramite."""
        pass

    def _initialize_pcmci(self, data):
        """Initialize Tigramite's PCMCI object with the prepared data and CI test."""
        pass

    def _run_pcmci(self, data, link_assumptions=None):
        """Run Tigramite's PCMCI procedure and return its results."""
        pass

    def _process_results(self, results):
        """Normalize Tigramite's results for pgmpy and TimeSeriesDAG."""
        pass

    @abstractmethod
    def _fit(self, X):
        """Fit the estimator and store the discovery results."""
        pass
