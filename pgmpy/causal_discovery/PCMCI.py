from pgmpy.causal_discovery._pcmci import _BasePCMCI


class PCMCI(_BasePCMCI):
    """
    Public estimator for PCMCI causal discovery.

    Accepts a sktime multi-index DataFrame and exposes discovery
    results for further conversion by TimeSeriesDAG.
    """

    def __init__(
        self,
        cond_ind_test,
        tau_min=0,
        tau_max=1,
        pc_alpha=0.05,
        fdr_method="fdr_bh",
        verbosity=0,
        link_assumptions=None,
        **kwargs,
    ):
        """Initialize the PCMCI estimator."""
        super().__init__(
            cond_ind_test=cond_ind_test,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            fdr_method=fdr_method,
            verbosity=verbosity,
            link_assumptions=link_assumptions,
            **kwargs,
        )

    def _fit(self, X):  # type: ignore
        """
        Prepare the input, run Tigramite's PCMCI procedure,
        and store the discovery results.
        """
        # Prepare the time-series data.
        data = self._prepare_data(X)

        # Run PCMCI using Tigramite.
        results = self._run_pcmci(
            data,
            link_assumptions=self.link_assumptions,
        )

        # Normalize and store the discovery results.
        results = self._process_results(results)

        self.p_matrix_ = results["p_matrix"]  # type: ignore
        self.val_matrix_ = results["val_matrix"]  # type: ignore
        self.graph_ = results["graph"]  # type: ignore
        self.candidates_ = results.get("candidates")  # type: ignore
        self.corrected_p_matrix_ = results.get("corrected_p_matrix")  # type: ignore

        return self
