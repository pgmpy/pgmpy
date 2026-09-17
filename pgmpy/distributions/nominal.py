import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from skpro.distributions.base import BaseDistribution


class NominalDistribution(BaseDistribution):
    """Nominal distribution for discrete random variables.

    Represents one or more nominal categorical probability distributions over a
    finite set of discrete states. Each row of ``probs`` defines the
    probability mass assigned to the states specified by ``categories``.

    The categories are treated as *nominal* (unordered, non-numeric) labels.
    Consequently, order- or arithmetic-based summaries are undefined and raise
    ``NotImplementedError``: ``cdf``, ``ppf``, ``mean``, ``var`` and ``energy``.
    The supported methods are ``pmf``, ``log_pmf``, ``sample`` and ``plot``.

    Parameters
    ----------
    probs : array-like of shape (n_instances, n_states)
        Probability masses for each state. Each row represents one
        nominal categorical distribution and must contain non-negative probs
        that sum to 1.
    categories : array-like of shape (n_states,)
        Names or labels of the possible discrete states. The order of
        ``categories`` corresponds to the order of probabilities in
        each row of ``probs``. State names must be unique.
    random_state : int, np.random.Generator, or None, optional, default = None
        Controls sampling in :meth:`sample`. An ``int`` produces reproducible
        draws that are identical on every call and across runs/processes. A
        ``numpy.random.Generator`` is used as-is, advancing across calls.
        ``None`` draws fresh, non-reproducible samples. Other methods are
        unaffected.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = ["variable"]

    Examples
    --------
    >>> from pgmpy.distributions.nominal import NominalDistribution

    >>> probs = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
    >>> categories = ["A", "B", "C", "D"]
    >>> index=["studentA", "studentB"]
    >>> columns = ["grade"]
    >>> dist = NominalDistribution(probs=probs, categories=categories, index=index, columns=columns)

    """

    _tags = {
        "python_version": None,
        "python_dependencies": None,
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "capabilities:approx": [],
        "capabilities:exact": ["pmf", "log_pmf"],
        "broadcast_init": "off",
    }

    def __init__(self, probs, categories, random_state=None, index=None, columns=None):

        self.probs = probs
        self.categories = categories
        self.random_state = random_state
        self.rng_ = np.random.default_rng(self.random_state)

        # Validate probs.
        probs_for_check = np.asarray(probs, dtype=float)
        if np.any(probs_for_check < 0):
            raise ValueError("probs must contain only non-negative probabilities")

        row_sums = probs_for_check.sum(axis=1)
        invalid_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=1e-9, atol=1e-12))
        if invalid_rows.size:
            raise ValueError(
                "The probabilities in each row of probs must sum to 1; "
                f"invalid row indices: {invalid_rows.tolist()}, "
                f"row sums: {row_sums[invalid_rows].tolist()}"
            )

        # Validate categories.
        expected_type = type(categories[0])
        mismatched_items = [
            (index, value, type(value).__name__)
            for index, value in enumerate(categories)
            if type(value) is not expected_type
        ]

        if mismatched_items:
            raise TypeError(
                "All categories must have the same type; "
                f"expected {expected_type.__name__}, "
                f"but found mismatched categories: {mismatched_items}"
            )

        # Validate shape of categories and probs.
        if len(categories) != len(set(categories)):
            raise ValueError(f"Categories must contain unique values: {categories}")

        if len(probs[0]) != len(categories):
            raise ValueError(
                f"""mismatch between the shape of categories and probs : {len(probs[0])}, {len(categories)}"""
            )

        # Validate index, columns.
        if index is None:
            index = pd.RangeIndex(len(probs))
        elif len(index) != len(probs):
            raise ValueError(f"The length of index must match the number of rows in probs : {len(index)}, {len(probs)}")
        if columns is None:
            columns = ["variable"]
        elif len(columns) != 1:
            raise ValueError("columns must contain exactly one column name")

        super().__init__(index=index, columns=columns)

    def _select_probs(self, x):
        """Look up the probability of each queried value.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        valid : 1D np.ndarray of bool
            Whether each queried value matches one of ``categories``.
        selected : 1D np.ndarray of float
            Probability of the matched category. The value at non-matching
            positions is arbitrary and is masked out by the callers.

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        matches = x == categories
        valid = matches.any(axis=1)
        state_idx = matches.argmax(axis=1)
        row_idx = np.arange(probs.shape[0])

        return valid, probs[row_idx, state_idx]

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        valid, selected = self._select_probs(x)
        res = np.where(valid, selected, 0.0)
        return res.reshape(-1, 1)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Values that do not match any category, or that match a
        zero-probability category, map to ``-inf``.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        valid, selected = self._select_probs(x)
        with np.errstate(divide="ignore"):
            log_selected = np.log(selected)
        res = np.where(valid, log_selected, -np.inf)
        return res.reshape(-1, 1)

    def cdf(self, x):
        """Not defined for a nominal categorical distribution.

        Categories have no inherent order, so the cumulative distribution
        function is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            "cdf is not defined for a nominal NominalDistribution: categories have no inherent order."
        )

    def ppf(self, p):
        """Not defined for a nominal categorical distribution.

        Categories have no inherent order, so the quantile (inverse-cdf)
        function is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            "ppf is not defined for a nominal NominalDistribution: categories have no inherent order."
        )

    def mean(self):
        """Not defined for a nominal categorical distribution.

        Categories are not numeric, so the expectation is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("mean is not defined for a nominal NominalDistribution: categories are not numeric.")

    def var(self):
        """Not defined for a nominal categorical distribution.

        Categories are not numeric, so the variance is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("var is not defined for a nominal NominalDistribution: categories are not numeric.")

    def energy(self, x=None):
        """Not defined for a nominal categorical distribution.

        Categories have no metric, so the energy distance is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("energy is not defined for a nominal NominalDistribution: categories have no metric.")

    def _sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        pd.DataFrame

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        if n_samples is None:
            n_samples = 1
            single_sample = True
        else:
            single_sample = False

        n_rows, _ = probs.shape

        sampled = np.empty((n_samples, n_rows), dtype=categories.dtype)

        for i in range(n_rows):
            sampled[:, i] = self.rng_.choice(
                categories,
                size=n_samples,
                p=probs[i],
                replace=True,
            )

        index = self.index
        columns = self.columns

        if single_sample:
            res = pd.DataFrame(
                sampled[0].reshape(-1, 1),
                index=index,
                columns=columns,
            )
        else:
            multi_index = pd.MultiIndex.from_product(
                [pd.RangeIndex(n_samples), index],
                names=["sample", None],
            )

            res = pd.DataFrame(
                sampled.reshape(n_samples * n_rows, 1),
                index=multi_index,
                columns=columns,
            )

        return res

    def plot(self, fun="pmf", ax=None, **kwargs):
        """Plot the nominal probability mass function.

        A separate bar plot is created for each row in ``probs``. The category
        labels are taken from ``categories``, and the height of each bar represents
        the corresponding probability.

        Each subplot is labeled using the corresponding entry in ``index``. The
        first entry in ``columns`` is used as the figure title.

        Parameters
        ----------
        fun : {"pmf"}, default="pmf"
            Distribution function to plot.
            Currently, only the probability mass function (``"pmf"``) is supported.
        ax : matplotlib Axes object, optional
            matplotlib Axes to plot in
            if not provided, defaults to current axes (``plot.gca``)
        kwargs : keyword arguments
            passed to the plotting function

        Returns
        -------
        fig : matplotlib.Figure, only returned if self is array distribution
            matplotlig Figure object for subplots
        ax : matplotlib.Axes
            the axis or axes on which the plot is drawn

        Notes
        -----
        The `matplotlib` library must be installed to use this method.

        Examples
        --------
        >>> probs = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
        >>> categories = ["A", "B", "C", "D"]
        >>> index = ["studentA", "studentB"]
        >>> columns = ["grade"]
        >>> dist = NominalDistribution(probs=probs, categories=categories, index=index, columns=columns)
        >>> dist.plot(fun="pmf") # doctest: +SKIP
        <Figure ...>
        """
        _check_soft_dependencies("matplotlib", obj="distribution plot")
        import matplotlib.pyplot as plt

        probs = np.asarray(self.probs, dtype=float)
        states = np.asarray(self.categories)

        n_rows, _ = probs.shape

        if fun != "pmf":
            raise NotImplementedError("`NominalDistribution` only supports `pmf` currently")

        if ax is None:
            fig, axes = plt.subplots(
                n_rows,
                1,
                squeeze=False,
                sharex=True,
                sharey=True,
            )
        else:
            axes = np.asarray(ax, dtype=object).reshape(n_rows, 1)
            fig = axes[0, 0].figure

        for i in range(n_rows):
            current_ax = axes[i, 0]
            current_ax.bar(states, probs[i], **kwargs)
            current_ax.set_ylabel(str(self.index[i]))
            current_ax.set_ylim(0, 1)

        axes[0, 0].set_title(str(self.columns[0]))
        axes[-1, 0].set_xlabel("state names")

        return fig, axes[:, 0]

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns.

        Returns
        -------
        dict

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        if rowidx is not None:
            probs = probs[rowidx, :]

            if probs.ndim == 1:
                probs = probs.reshape(1, -1)

            if categories.ndim == 2:
                categories = categories[rowidx, :]
                if categories.ndim == 1:
                    categories = categories.reshape(1, -1)

        return {
            "probs": probs,
            "categories": categories,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"probs": [[0.1, 0.9], [0.7, 0.3]], "categories": [1, 2]}
        params2 = {"probs": [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], "categories": [1, 2, 3]}
        return [params1, params2]
