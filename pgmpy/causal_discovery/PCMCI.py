from pgmpy.causal_discovery._base_pcmpi import _BasePCMCI


class PCMCI(_BasePCMCI):
    """
    Discover causal links in one or more time series using Tigramite PCMCI.

    Parameters
    ----------
    cond_ind_test : object
        Native Tigramite conditional independence test.

    tau_min : int, default=0
        Minimum time lag considered during causal discovery.

    tau_max : int, default=1
        Maximum time lag considered during causal discovery.

    pc_alpha : float, default=0.05
        Significance level for the PC-stable phase.

    fdr_method : str or None, default="fdr_bh"
        Multiple-testing correction method.

    verbosity : int, default=0
        Tigramite verbosity level.

    link_assumptions : dict or None, default=None
        Optional restrictions on candidate causal links.

    **kwargs
        Additional arguments passed to Tigramite's ``run_pcmci``.

    Attributes
    ----------
    graph_ : numpy.ndarray
        Discovered causal graph.

    p_matrix_ : numpy.ndarray
        Matrix of p-values for candidate links.

    val_matrix_ : numpy.ndarray
        Matrix of test statistics for candidate links.

    candidates_ : object
        Candidate links returned by Tigramite, if available.

    corrected_p_matrix_ : numpy.ndarray or None
        Multiple-testing-corrected p-values, if returned by Tigramite.
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
