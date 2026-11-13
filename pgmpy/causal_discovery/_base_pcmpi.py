import numpy as np

from pgmpy.causal_discovery._baseTimeSeries import BaseTimeSeriesCausalDiscovery


class _BasePCMCI(BaseTimeSeriesCausalDiscovery):
    """
    Implement the Tigramite-specific steps of time-series discovery.

    Shared input validation and fit coordination are provided by
    :class:`BaseTimeSeriesCausalDiscovery`.
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
        """Initialize PCMCI options and the shared time-series estimator."""
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

        self.tigramite_data_ = None
        self.pcmci_ = None

        self.candidates_ = None
        self.corrected_p_matrix_ = None

    def _initialize_algorithm(self) -> None:
        """Build Tigramite's data frame and initialize its PCMCI engine."""
        from tigramite import data_processing as dp
        from tigramite.pcmci import PCMCI as TigramitePCMCI

        if self.series_data_ is None:
            raise RuntimeError("Time-series data must be prepared before initializing the algorithm.")

        if self.tigramite_ci_test_ is None:
            raise RuntimeError("The conditional independence test must be adapted before initializing PCMCI.")

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
        """Validate link assumptions and normalize their indices for Tigramite."""
        assumptions = self.link_assumptions

        if assumptions is None:
            return None

        if not isinstance(assumptions, dict):
            raise TypeError("link_assumptions must be a dictionary or None.")

        if self.n_variables_ is None:
            raise RuntimeError("Prepare time-series data before validating link assumptions.")

        normalized = {}
        allowed_marks = {"->", "<-", "oo", "?>", "<?"}

        for target, links in assumptions.items():
            if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)):
                raise TypeError("Target variable indices in link_assumptions must be integers.")

            if not 0 <= target < self.n_variables_:
                raise ValueError(f"Target variable index {target} is out of valid range [0, {self.n_variables_ - 1}].")

            if not isinstance(links, dict):
                raise TypeError(f"Link assumptions for target variable {target} must be a dictionary.")

            normalized_links = {}

            for link, mark in links.items():
                if not isinstance(link, tuple) or len(link) != 2:
                    raise TypeError("Each link key must be a 2-tuple: (source_variable_index, time_lag).")

                source, lag = link

                if not isinstance(source, (int, np.integer)) or isinstance(source, (bool, np.bool_)):
                    raise TypeError("Source variable indices must be integers.")

                if not 0 <= source < self.n_variables_:
                    raise ValueError(
                        f"Source variable index {source} is out of valid range [0, {self.n_variables_ - 1}]."
                    )

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

    def _run_discovery(self) -> dict[str, object]:
        """Run PCMCI and return Tigramite's raw results."""
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
        """Validate and store the graph, p-values, and test statistics."""
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
