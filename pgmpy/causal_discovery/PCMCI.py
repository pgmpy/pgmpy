from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery._pcmci import _BasePCMCI


class PCMCI(BaseCausalDiscovery, _BasePCMCI):
    """
    Public class which accepts a pandas dataframe
    and exposes statistical discovery results to ]
    TimeSeriesDAG for further conversion.
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
        _BasePCMCI.__init__(
            self,
            cond_ind_test=cond_ind_test,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            fdr_method=fdr_method,
            verbosity=verbosity,
            link_assumptions=link_assumptions,
            **kwargs,
        )

    def _fit(self, X):
        """
        Fit the PCMCI causal discovery algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            Time-series data.

        Returns
        -------
        self
            Fitted PCMCI estimator.
        """
        pass


"""
Result coming out of _run_pcmci():
result = {
    "p_matrix": p_matrix,
    "val_matrix": val_matrix,
    "graph": graph,
    "candidates": candidates,
    "corrected_p_matrix": corrected_p_matrix,
}.
"""
