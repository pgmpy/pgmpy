from copy import deepcopy

import numpy as np

from pgmpy.causal_discovery._baseTimeSeries import BaseTimeSeriesCausalDiscovery


class _BasePCMCI(BaseTimeSeriesCausalDiscovery):
    """
    Base estimator for causal discovery algorithms built on top of
    Tigramite's PCMCI framework.

    This class bridges the generic time-series causal discovery lifecycle
    with Tigramite. Concrete subclasses define their own public estimator
    interfaces while inheriting the core discovery and validation logic here.
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
        """
        Initialize the PCMCI base estimator.

        Parameters
        ----------
        cond_ind_test : object
            Conditional independence test wrapper compatible with TigramiteAdapter.

        tau_min : int, default=0
            Minimum time lag considered during discovery.

        tau_max : int, default=1
            Maximum time lag considered during discovery.

        pc_alpha : float, default=0.05
            Significance level used for the PC-stable conditional independence tests.

        fdr_method : str or None, default="fdr_bh"
            Multiple-testing correction method applied by Tigramite.

        verbosity : int, default=0
            Verbosity level forwarded to Tigramite execution logs.

        link_assumptions : dict or None, default=None
            Optional dictionary specifying prior domain assumptions on links.

        **kwargs : dict
            Additional keyword arguments passed directly to Tigramite's `run_pcmci()`.
        """
        super().__init__(
            tau_min=tau_min,
            tau_max=tau_max,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity,
        )

        if not isinstance(pc_alpha, (int, float)) or isinstance(pc_alpha, bool):
            raise TypeError("pc_alpha must be a numeric value.")

        if not 0 < pc_alpha < 1:
            raise ValueError("pc_alpha must be strictly between 0 and 1.")

        self.pc_alpha = pc_alpha
        self.fdr_method = fdr_method
        self.link_assumptions = link_assumptions
        self.kwargs = kwargs

        # Internal Tigramite state references
        self.tigramite_ci_test_ = None
        self.tigramite_data_ = None
        self.pcmci_ = None

        # Discovered causal outputs
        self.candidates_ = None
        self.corrected_p_matrix_ = None

    def _initialize_algorithm(self) -> None:
        """
        Convert prepared internal series data into Tigramite's DataFrame
        representation and instantiate the PCMCI engine.
        """
        from tigramite import data_processing as dp
        from tigramite.pcmci import PCMCI as TigramitePCMCI

        if self.series_data_ is None:
            raise RuntimeError("Time-series data must be prepared before initializing the algorithm.")

        if self.tigramite_ci_test_ is None:
            raise RuntimeError("The conditional independence test must be adapted before initializing PCMCI.")

        # Map each variable index to a float NumPy array matching variable_names_
        data = {index: series.to_numpy(dtype=float) for index, series in enumerate(self.series_data_.values())}

        self.tigramite_data_ = dp.DataFrame(
            data=data,
            var_names=self.variable_names_,
        )

        self.pcmci_ = TigramitePCMCI(
            dataframe=self.tigramite_data_,
            cond_ind_test=self.tigramite_ci_test_,
            verbosity=self.verbosity,
        )

    def _set_link_assumptions(self):
        """
        Validate and normalize optional user-defined link assumptions.

        Returns
        -------
        dict or None
            A validated dictionary of link assumptions formatted for Tigramite.
        """
        assumptions = self.link_assumptions

        if assumptions is None:
            return None

        if not isinstance(assumptions, dict):
            raise TypeError("link_assumptions must be a dictionary or None.")

        normalized = {}
        allowed_marks = {"->", "<-", "oo", "?>", "<?"}

        for target, links in assumptions.items():
            if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)):
                raise TypeError("Target variable indices in link_assumptions must be integers.")

            if not 0 <= target < self.n_variables:  # type: ignore
                raise ValueError(
                    f"Target variable index {target} is out of valid range [0, {self.n_variables_ - 1})."  # type: ignore
                )

            if not isinstance(links, dict):
                raise TypeError(f"Link assumptions for target variable {target} must be provided as a dictionary.")

            normalized_links = {}

            for link, mark in links.items():
                if not isinstance(link, tuple) or len(link) != 2:
                    raise TypeError("Each link key must be a 2-tuple: (source_variable_index, time_lag).")

                source, lag = link

                if not isinstance(source, (int, np.integer)) or isinstance(source, (bool, np.bool_)):
                    raise TypeError("Source variable indices must be integers.")

                if not 0 <= source < self.n_variables:  # type: ignore
                    raise ValueError(f"Source variable index {source} is out of valid range.")

                if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)):
                    raise TypeError("Time lags must be integer values.")

                if not -self.tau_max <= lag <= -self.tau_min:
                    raise ValueError(f"Time lag {lag} must fall between {-self.tau_max} and {-self.tau_min}.")

                if source == target and lag == 0:
                    raise ValueError("Contemporaneous self-links (lag 0 on the same variable) are not allowed.")

                if mark not in allowed_marks:
                    raise ValueError(f"Invalid link mark {mark!r}. Expected one of: {sorted(allowed_marks)}.")

                normalized_links[(int(source), int(lag))] = mark

            normalized[int(target)] = normalized_links

        return normalized

    def _wrap_ci_test_for_tigramite(self):
        """Prepare a native Tigramite CI test for PCMCI."""
        if self.cond_ind_test is None:
            raise ValueError("A Tigramite conditional-independence test must be provided.")

        if not callable(getattr(self.cond_ind_test, "set_dataframe", None)):
            raise TypeError("PCMCI requires a native Tigramite CI test exposing 'set_dataframe()'.")

        self.tigramite_ci_test_ = deepcopy(self.cond_ind_test)
        return self.tigramite_ci_test_

    def _run_discovery(self) -> dict[str, object]:
        """
        Run Tigramite's PCMCI causal discovery.

        Returns
        -------
        dict
            Raw dictionary returned by Tigramite's run_pcmci().
        """
        if self.pcmci_ is None:
            raise RuntimeError("PCMCI must be initialized before running causal discovery.")

        run_kwargs = {
            **self.kwargs,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "fdr_method": self.fdr_method,
            "link_assumptions": self._set_link_assumptions(),
        }

        return self.pcmci_.run_pcmci(**run_kwargs)

    def _process_discovery_results(
        self,
        results: dict[str, object],
    ) -> None:
        """
        Validate, parse, and store the results returned by Tigramite.

        Ensures graphs, p-values, and test statistics conform to the expected
        shape: (n_variables, n_variables, tau_max + 1).
        """
        if not isinstance(results, dict):
            raise TypeError("Discovery results must be returned as a dictionary.")

        required_keys = {"graph", "p_matrix", "val_matrix"}
        missing_keys = required_keys - results.keys()

        if missing_keys:
            raise ValueError(f"Tigramite results are missing required keys: {sorted(missing_keys)}.")

        expected_shape = (
            self.n_variables_,
            self.n_variables_,
            self.tau_max + 1,
        )

        graph = np.asarray(results["graph"], dtype=str)
        p_matrix = np.asarray(results["p_matrix"], dtype=float)
        val_matrix = np.asarray(results["val_matrix"], dtype=float)

        for name, matrix in (
            ("graph", graph),
            ("p_matrix", p_matrix),
            ("val_matrix", val_matrix),
        ):
            if matrix.shape != expected_shape:
                raise ValueError(f"Tigramite's {name} has shape {matrix.shape}; expected shape {expected_shape}.")

        self.graph_ = graph
        self.p_matrix_ = p_matrix
        self.val_matrix_ = val_matrix
        self.candidates_ = results.get("candidates")

        corrected_p_matrix = results.get("corrected_p_matrix")
        if corrected_p_matrix is not None:
            corrected_p_matrix = np.asarray(corrected_p_matrix, dtype=float)

            if corrected_p_matrix.shape != expected_shape:
                raise ValueError(
                    f"Tigramite's corrected_p_matrix has shape {corrected_p_matrix.shape}; expected {expected_shape}."
                )

        self.corrected_p_matrix_ = corrected_p_matrix
