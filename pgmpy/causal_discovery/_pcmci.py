from abc import ABC


class _BasePCMCI(ABC):
    """Base class containing the common engine for PCMCI algorithm(s)

    This class handles:
    - PCMCI parameter handling
    - CI test handling
    - lag/link assumption
    - candidate-parent initialization
    - PC / condition-selection phase
    - MCI Phase
    - p-value correction
    - assembling PCMCI discovery results

    The resulting matrices are feeded to TimeSeriesDAG layer for conversion.
    """

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
        self.cond_ind_test = cond_ind_test
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.fdr_method = fdr_method
        self.verbosity = verbosity
        self.link_assumptions = link_assumptions

    # Data prep
    def _prepare_data(self, X):
        """prepate validated pgmpy dataframe for PCMCI internals"""
        pass

    def _run_ci_tests(self, data, source, target, condition_set):
        """Run 1 CI test

        Returns
        -------
        val : float
            Test statistic / dependency measure

        p_value : float
            p-value returned by the CI test.
        """
        pass

    def _set_link_assumptions(self, link_assumptions=None):
        """Validate and prepare temporal links

        If no assumptions are provided, build the default
        set tuple of source-target-lag.
        """
        pass

    def _initialize_candidates(self):
        """Initialize candidate parents for every target variable.

        Returns
        -------
        candidates : dict
            Mapping from target variable to candidate.
        """
        pass

    def _run_pc_stable(
        self,
        data,
        candidates,
    ):
        """Run PC stable / condition-selection phase of PCMCI.

        Returns
        -------
        candidates : dict
            Remaining candidate parent relationships after
            conditional independence based pruning.
        """
        pass

    def _build_condition_set(
        self,
        source,
        target,
        candidates,
        condition_set_size,
    ):
        """Construct a conditioning parent for a PC-phase CI test."""
        pass

    def _build_mci_condition_set(self, source, target, candidates):
        """Condition set using MCI test."""
        pass

    def _run_mci_tests(
        self,
        data,
        candidates,
    ):
        """Run MCI tests for all candidate relationships

        Returns
        -------
        p_matrix : np.ndarray
            Temporal p-value matrix

        val_matrix : np.ndarray
            Temporal test-statistic / dependency-value matrix.
        """
        pass

    def _run_pcmci(self, data, link_assumptions=None):
        """Execute the complete PCMCI workflow

        Steps:
        - Configure link assumptions
        - Initialize candidate parents
        - Run PC / condition-selection phase
        - Run MCI tests
        - correct p-values
        - Determine discovered links

        Returns
        -------
        result : dict
            PCMCI discovery results.
        """
        pass
