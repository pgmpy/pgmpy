# Adapter from Tigramite to pgmpy for CI test

import numpy as np
import pandas as pd

from pgmpy.ci_tests._base import BaseCITest


class PgmpyTigramiteCITest:
    """Adapt a pgmpy CI test to Tigramite's conditional-independence interface.

    Parameters
    ----------
    cond_ind_test : BaseCITest
        An instantiated pgmpy conditional independence test.

    variable_names : list
        Names of the variables in the original time-series dataset.
    """

    def __init__(self, cond_ind_test, variable_names):
        if not isinstance(cond_ind_test, BaseCITest):
            raise TypeError("cond_ind_test must be an instance of pgmpy.ci_tests.BaseCITest.")

        if not variable_names:
            raise ValueError("variable_names must contain at least one variable.")

        self.cond_ind_test = cond_ind_test
        self.variable_names = list(variable_names)
        self.dataframe = None

    @property
    def measure(self):
        """Identify the underlying pgmpy test for Tigramite."""
        return f"pgmpy_{type(self.cond_ind_test).__name__}"

    def set_dataframe(self, dataframe):
        """Store the Tigramite DataFrame and validate its variables."""
        if not hasattr(dataframe, "values"):
            raise TypeError("dataframe must be a Tigramite DataFrame with a values attribute.")

        values = dataframe.values

        # Tigramite can represent one or more independent series as a dict.
        if isinstance(values, dict):
            series = values.values()
        else:
            series = [values]

        for data in series:
            data = np.asarray(data)

            if data.ndim != 2:
                raise ValueError("Each time series must have shape (observations, variables).")

            if data.shape[1] != len(self.variable_names):
                raise ValueError("The number of columns in the Tigramite data does not match variable_names.")

        self.dataframe = dataframe

    def set_mask_type(self, mask_type):
        """Reject masking until mask support is implemented."""
        if mask_type is not None:
            raise NotImplementedError("This initial adapter does not support Tigramite masking.")

    def _normalise_nodes(self, nodes, name):
        """Validate Tigramite nodes represented as (variable, lag) pairs."""
        if nodes is None:
            return []

        # Tigramite usually supplies lists such as [(0, -1)].
        if isinstance(nodes, tuple) and len(nodes) == 2:
            nodes = [nodes]

        normalized = []

        for node in nodes:
            if not isinstance(node, (tuple, list)) or len(node) != 2:
                raise ValueError(f"{name} must contain (variable_index, lag) pairs.")

            variable, lag = node

            if (
                not isinstance(variable, (int, np.integer))
                or isinstance(variable, (bool, np.bool_))
                or not 0 <= variable < len(self.variable_names)
            ):
                raise ValueError(f"Invalid variable index in {name}: {variable!r}")

            if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)) or lag > 0:
                raise ValueError(f"Invalid lag in {name}: {lag!r}. Expected zero or a negative integer.")

            normalized.append((int(variable), int(lag)))

        return normalized

    def _build_lagged_data(self, X, Y, Z, tau_max, cut_off):
        """Build aligned columns for the variables involved in one CI test."""
        if self.dataframe is None:
            raise RuntimeError("set_dataframe() must be called before run_test().")

        nodes = list(dict.fromkeys(X + Y + Z))

        if not nodes:
            raise ValueError("At least one variable must be supplied.")

        max_requested_lag = max((-lag for _, lag in nodes), default=0)

        if cut_off == "2xtau_max":
            cutoff = 2 * tau_max
        elif cut_off == "max_lag":
            cutoff = max_requested_lag
        elif cut_off == "max_lag_or_tau_max":
            cutoff = max(max_requested_lag, tau_max)
        elif isinstance(cut_off, (int, np.integer)) and cut_off >= 0:
            cutoff = int(cut_off)
        else:
            raise ValueError("cut_off must be '2xtau_max', 'max_lag', 'max_lag_or_tau_max', or a non-negative integer.")

        cutoff = max(cutoff, max_requested_lag)

        values = self.dataframe.values
        series = list(values.values()) if isinstance(values, dict) else [values]

        columns = {node: f"var_{node[0]}_lag_{node[1]}" for node in nodes}
        aligned = {columns[node]: [] for node in nodes}

        for data in series:
            data = np.asarray(data)

            if data.ndim != 2:
                raise ValueError("Time-series data must be two-dimensional.")

            if data.shape[1] != len(self.variable_names):
                raise ValueError("Unexpected number of variables in time-series data.")

            if data.shape[0] <= cutoff:
                raise ValueError("A time series is too short for the requested lag/cutoff.")

            # t is the current time; a node with lag -1 reads t - 1.
            times = np.arange(cutoff, data.shape[0])

            for variable, lag in nodes:
                aligned[columns[(variable, lag)]].append(data[times + lag, variable])

        lagged_data = pd.DataFrame({name: np.concatenate(chunks) for name, chunks in aligned.items()})

        if lagged_data.empty:
            raise ValueError("No aligned observations remain after applying lags.")

        if not np.isfinite(lagged_data.to_numpy(dtype=float)).all():
            raise ValueError("Aligned data contains missing or non-finite values.")

        return lagged_data, columns

    def run_test(
        self,
        X,
        Y,
        Z=None,
        tau_max=0,
        cut_off="2xtau_max",
        alpha_or_thres=None,
        **kwargs,
    ):
        """Run a pgmpy CI test using Tigramite's lagged-node representation.

        Returns
        -------
        tuple
            ``(statistic, p_value)`` or
            ``(statistic, p_value, dependent)`` when alpha_or_thres is given.
        """
        X = self._normalise_nodes(X, "X")
        Y = self._normalise_nodes(Y, "Y")
        Z = self._normalise_nodes(Z, "Z")

        if len(X) != 1 or len(Y) != 1:
            raise ValueError("This adapter currently supports one X node and one Y node per test.")

        if not isinstance(tau_max, (int, np.integer)) or tau_max < 0:
            raise ValueError("tau_max must be a non-negative integer.")

        if alpha_or_thres is not None and not 0 <= alpha_or_thres <= 1:
            raise ValueError("alpha_or_thres must be between 0 and 1.")

        lagged_data, columns = self._build_lagged_data(
            X=X,
            Y=Y,
            Z=Z,
            tau_max=int(tau_max),
            cut_off=cut_off,
        )

        x_name = columns[X[0]]
        y_name = columns[Y[0]]
        z_names = [columns[node] for node in Z]

        # pgmpy's CI tests operate on named columns of a DataFrame.
        # Rebind the test to this query's lag-aligned data.
        self.cond_ind_test.data = lagged_data  # type: ignore
        self.cond_ind_test.dtypes = dict.fromkeys(  # type: ignore
            lagged_data.columns, "N"
        )

        # The same names can have different results
        # due to cutoff settings.
        if hasattr(self.cond_ind_test, "_result_cache"):
            self.cond_ind_test._result_cache.clear()

        statistic, p_value = self.cond_ind_test.run_test(
            X=x_name,
            Y=y_name,
            Z=z_names,
        )

        if statistic is None or not np.isfinite(statistic):
            raise ValueError("The pgmpy CI test must return a finite numeric statistic.")

        if p_value is None or not np.isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError("The pgmpy CI test must return a valid p-value between 0 and 1.")

        statistic = float(statistic)
        p_value = float(p_value)

        if alpha_or_thres is not None:
            # Tigramite uses dependent=True to mean reject independence.
            dependent = p_value < alpha_or_thres
            return statistic, p_value, dependent

        return statistic, p_value
