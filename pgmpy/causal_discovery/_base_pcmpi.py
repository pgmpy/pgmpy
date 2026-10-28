from abc import abstractmethod

import numpy as np

from pgmpy.causal_discovery._baseTimeSeries import (
    BaseTimeSeriesCausalDiscovery,
)
from pgmpy.ci_tests._tigramite_adapter import TigramiteAdapter


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
        super().__init__(
            tau_min=tau_min,
            tau_max=tau_max,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity,
        )

        if not 0 < pc_alpha < 1:
            raise ValueError("pc_alpha must be between 0 and 1.")

        self.pc_alpha = pc_alpha
        self.fdr_method = fdr_method
        self.link_assumptions = link_assumptions
        self.kwargs = kwargs
        self.tigramite_ci_test_ = None
        self.ci_test_ = None
        self.pcmci_ = None

    def _prepare_data(self, X):
        """Convert an sktime MultiIndex DataFrame to a Tigramite DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        tigramite_data_ : tigtamite DataFrame
            A multi-indexed tigramite friendly dataframe.
        """

        from tigramite import data_processing as dp

        self._validate_time_series_data(X)
        self.X_ = X.copy()
        self.series_data_ = self._prepare_time_series_data(X)

        too_short = [series_id for series_id, series in self.series_data_.items() if len(series) <= self.tau_max]

        if too_short:
            raise ValueError(
                f"Each time series must contain more observation than tau_max. Series that are too short: {too_short!r}"
            )

        data = {i: series.to_numpy(dtype=float) for i, series in enumerate(self.series_data_.values())}

        self.tigramite_data_ = dp.DataFrame(data=data, var_names=self.variable_names_)

        return self.tigramite_data_

    def _wrap_ci_test_for_tigramite(self):
        """Create the adapter for pgmpy's conditional independence test."""

        if self.cond_ind_test is None:
            raise ValueError("cond_ind_test must be provided for causal discovery.")

        from copy import deepcopy

        self.ci_test_ = deepcopy(self.cond_ind_test)

        self.tigramite_ci_test_ = TigramiteAdapter(
            tigramite_ci=self.ci_test_, tau_max=self.tau_max, cut_off="2xtau_max", alpha_or_thres=None
        )

        return self.tigramite_ci_test_

    def _set_link_assumptions(self, link_assumptions=None):
        """Validate link assumptions in Tigramite's native index format."""
        if link_assumptions is None:
            link_assumptions = self.link_assumptions

        if link_assumptions is None:
            return None

        if not isinstance(link_assumptions, dict):
            raise TypeError("link_assumptions must be a dictionary.")

        n_variables = self.n_variables_
        allowed_marks = {"->", "<-", "oo", "?>", "<?"}
        normalized = {}

        for target, assumptions in link_assumptions.items():
            if (
                not isinstance(target, (int, np.integer))
                or isinstance(target, (bool, np.bool_))
                or not 0 <= target < n_variables  # type: ignore
            ):
                raise ValueError(
                    f"Invalid target index {target!r}; expected an integer between 0 and {n_variables - 1}."  # type: ignore
                )

            if not isinstance(assumptions, dict):
                raise TypeError(f"Link assumptions for target {target} must be a dictionary.")

            normalized[int(target)] = {}

            for link, mark in assumptions.items():
                if (
                    not isinstance(link, tuple)
                    or len(link) != 2
                    or not all(
                        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                        for value in link
                    )
                ):
                    raise ValueError("Each link must be a tuple (source_index, lag).")

                source, lag = map(int, link)

                if not 0 <= source < n_variables:  # type: ignore
                    raise ValueError(
                        f"Invalid source index {source}; expected an index between 0 and {n_variables - 1}."  # type: ignore
                    )

                # Tigramite represents past lags as negative integers.
                min_lag = -self.tau_max
                max_lag = -self.tau_min

                if not min_lag <= lag <= max_lag:
                    raise ValueError(f"Invalid lag {lag}; expected a lag between {min_lag} and {max_lag}")

                if lag == 0 and source == int(target):
                    raise ValueError("Self-links at lag 0 are not valid link assumptions.")

                if mark not in allowed_marks:
                    raise ValueError(f"Invalid link mark {mark!r}. Expected one of {sorted(allowed_marks)}.")

                normalized[int(target)][(source, lag)] = mark

        return normalized

    def _initialize_pcmci(self, data):
        """Initialize Tigramite's PCMCI object."""
        from tigramite.pcmci import PCMCI as TigramitePCMCI

        if self.tigramite_ci_test_ is None:
            self._wrap_ci_test_for_tigramite()

        self.pcmci_ = TigramitePCMCI(
            dataframe=data,
            cond_ind_test=self.tigramite_ci_test_,
            verbosity=self.verbosity,
        )

        return self.pcmci_

    def _run_pcmci(self, data, link_assumptions=None):
        """Run Tigramite's PCMCI algorithm."""
        self._initialize_pcmci(data=data)

        assumptions = self._set_link_assumptions(link_assumptions=link_assumptions)

        run_kwargs = {
            **self.kwargs,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "fdr_method": self.fdr_method,
            "link_assumptions": assumptions,
        }

        return self.pcmci_.run_pcmci(**run_kwargs)  # type: ignore

    def _process_results(self, results):
        """Validate and normalize Tigramite's result matrices."""
        required = {"graph", "p_matrix", "val_matrix"}
        missing = required - results.keys()

        if missing:
            raise ValueError(f"Tigramite results are missing required keys: {sorted(missing)}")

        expected_shape = (
            self.n_variables_,
            self.n_variables_,
            self.tau_max + 1,
        )

        normalized = dict(results)

        for name in required:
            dtype = str if name == "graph" else float
            matrix = np.asarray(results[name], dtype=dtype)

            if matrix.shape != expected_shape:
                raise ValueError(f"Tigramite returned {name} with shape {matrix.shape}; expected {expected_shape}")

            normalized[name] = matrix

        return normalized

    @abstractmethod
    def _fit(self, X):
        """Fit the estimator and store the discovery results."""
        ...
