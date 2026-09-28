from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

BaseDistribution = _safe_import("skpro.distributions.base.BaseDistribution")

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class NominalDistribution(BaseDistribution):
    """Nominal distribution for discrete random variables.

    Represents one or more nominal categorical probability distributions over a finite set of discrete states. A
    one-dimensional ``probs`` vector without ``index`` or ``columns`` defines a scalar distribution with shape ``()``. A
    two-dimensional ``probs`` array defines a distribution of shape ``(n_instances, 1)``, with each row assigning
    probability mass to the states specified by ``categories``.

    The categories are treated as *nominal* (unordered, non-numeric) labels. Consequently, order-, arithmetic- and
    density-based summaries are undefined and raise ``NotImplementedError``: ``cdf``, ``ppf``, ``surv``, ``haz``,
    ``mean``, ``var``, ``energy`` and ``pdfnorm``. The supported methods are ``pmf``, ``log_pmf``, ``sample`` and
    ``plot``.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    probs : array-like of shape (n_states,) or (n_instances, n_states)
        Probability masses for each state. Each vector represents one nominal categorical distribution and must contain
        non-negative probabilities that sum to 1, within an absolute tolerance of 0.01 (as for ``TabularCPD``). Each
        vector is rescaled to sum to exactly 1. A one-dimensional vector with an explicit ``index`` or ``columns``
        represents a single-row array distribution.
    categories : array-like of shape (n_states,)
        Labels of the possible discrete states. The order of ``categories`` corresponds to the order of probabilities in
        each row of ``probs``. Categories can be any distinct, hashable values, including values of different types, but
        not missing values.
    index : pd.Index or list, optional, default = None
        Row labels for an array distribution, defaulting to a ``RangeIndex``. Scalar distributions have no row labels.
    columns : pd.Index or list, optional, default = None
        One column label for an array distribution, defaulting to ``["variable"]``. Scalar distributions have no column
        labels.

    Examples
    --------
    >>> from pgmpy.parameterization.distributions import NominalDistribution
    >>> scalar = NominalDistribution(probs=[0.2, 0.8], categories=["A", "B"])
    >>> scalar.shape
    ()
    >>> float(scalar.pmf("A"))
    0.2

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
        "distr:paramtype": "nonparametric",
        "capabilities:approx": [],
        "capabilities:exact": ["pmf", "log_pmf"],
        "broadcast_init": "off",
    }

    def __init__(
        self,
        probs: ArrayLike,
        categories: ArrayLike,
        index: pd.Index | list | None = None,
        columns: pd.Index | list | None = None,
    ) -> None:
        _check_soft_dependencies("skpro", obj=self)

        self.probs = probs
        self.categories = categories

        # Validate probs.
        probs_arr = np.asarray(probs, dtype=float)
        if probs_arr.ndim not in (1, 2):
            raise ValueError("probs must be a one- or two-dimensional array")
        is_scalar = probs_arr.ndim == 1 and index is None and columns is None
        probs_arr = np.atleast_2d(probs_arr)
        if np.any(probs_arr < 0):
            raise ValueError("probs must contain only non-negative probabilities")

        # Same tolerance as TabularCPD.
        row_sums = probs_arr.sum(axis=1)
        invalid_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=0.01))
        if invalid_rows.size:
            raise ValueError(
                f"The probabilities in each row of probs must sum to 1; invalid row indices: {invalid_rows.tolist()}, "
                f"row sums: {row_sums[invalid_rows].tolist()}"
            )

        # Validate categories. An object array keeps each label as given, including tuples and mixed types.
        if isinstance(categories, str):
            raise TypeError(f"categories must be an array-like of labels, not a single string: {categories!r}")
        categories_arr = np.fromiter(categories, dtype=object)
        if pd.isna(categories_arr).any():
            raise ValueError(f"Categories must not contain missing values: {categories}")

        # Validate shape of categories and probs.
        if len(categories_arr) != len(set(categories_arr)):
            raise ValueError(f"Categories must contain unique values: {categories}")

        if probs_arr.shape[1] != len(categories_arr):
            raise ValueError(
                f"mismatch between the shape of categories and probs : {len(categories_arr)}, {probs_arr.shape[1]}"
            )

        # Validate index, columns.
        if not is_scalar:
            n_rows = probs_arr.shape[0]
            if index is None:
                index = pd.RangeIndex(n_rows)
            elif len(index) != n_rows:
                raise ValueError(f"The length of index must match the number of rows in probs : {len(index)}, {n_rows}")
            if columns is None:
                columns = ["variable"]
            elif len(columns) != 1:
                raise ValueError("columns must contain exactly one column name")

        self._probs = probs_arr / row_sums[:, np.newaxis]
        self._categories = categories_arr

        super().__init__(index=index, columns=columns)

    def _select_probs(self, x):
        """Look up the probability of each queried value.

        Parameters
        ----------
        x : scalar or 2D np.ndarray

        Returns
        -------
        valid : 1D np.ndarray of bool
            Whether each queried value matches one of ``categories``.
        selected : 1D np.ndarray of float
            Probability of the matched category. The value at non-matching positions is arbitrary and is masked out by
            the callers.

        """
        x = np.atleast_2d(x)
        if pd.isna(x).any():
            raise ValueError("Queries must not contain missing values (None, NaN or pd.NA)")

        matches = x == self._categories
        valid = matches.any(axis=1)
        state_idx = matches.argmax(axis=1)
        row_idx = np.arange(self._probs.shape[0])

        return valid, self._probs[row_idx, state_idx]

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : scalar or 2D np.ndarray

        Returns
        -------
        0D or 2D np.ndarray

        """
        valid, selected = self._select_probs(x)
        res = np.where(valid, selected, 0.0)
        return res.reshape(self.shape)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Values that do not match any category, or that match a zero-probability category, map to ``-inf``.

        Parameters
        ----------
        x : scalar or 2D np.ndarray

        Returns
        -------
        0D or 2D np.ndarray

        """
        valid, selected = self._select_probs(x)
        with np.errstate(divide="ignore"):
            log_selected = np.log(selected)
        res = np.where(valid, log_selected, -np.inf)
        return res.reshape(self.shape)

    def cdf(self, x: ArrayLike) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories have no inherent order, so the cumulative distribution function is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("cdf is not defined for NominalDistribution: categories have no inherent order.")

    def ppf(self, p: ArrayLike) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories have no inherent order, so the quantile (inverse-cdf) function is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("ppf is not defined for NominalDistribution: categories have no inherent order.")

    def surv(self, x: ArrayLike) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories have no inherent order, so the survival function is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("surv is not defined for NominalDistribution: categories have no inherent order.")

    def haz(self, x: ArrayLike) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories have no inherent order, so the hazard function is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("haz is not defined for NominalDistribution: categories have no inherent order.")

    def mean(self) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories are not numeric, so the expectation is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("mean is not defined for NominalDistribution: categories are not numeric.")

    def var(self) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories are not numeric, so the variance is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("var is not defined for NominalDistribution: categories are not numeric.")

    def energy(self, x: ArrayLike | None = None) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        Categories have no metric, so the energy distance is undefined.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("energy is not defined for NominalDistribution: categories have no metric.")

    def pdfnorm(self, a: float = 2) -> NoReturn:
        """Not defined for a nominal categorical distribution.

        The pdf norm is defined through a probability density, which a discrete distribution does not have.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("pdfnorm is not defined for NominalDistribution: it has no probability density.")

    def sample(self, n_samples: int | None = None, random_state: int | np.random.Generator | None = None) -> Any:
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            Number of samples to draw. ``None`` draws a single sample.
        random_state : int, np.random.Generator, or None, optional, default = None
            Controls the randomness of this call. An ``int`` produces the same draws on every call. A
            ``numpy.random.Generator`` is used as-is, advancing across calls. ``None`` draws fresh, non-reproducible
            samples.

        Returns
        -------
        scalar or pd.DataFrame
            A scalar distribution returns one category when ``n_samples`` is ``None``, and otherwise a ``pd.DataFrame``
            with one row per sample. An array distribution returns a ``pd.DataFrame`` with the index and columns of
            ``self``; multiple samples add an unnamed first index level numbering the samples.

        Examples
        --------
        >>> dist = NominalDistribution(probs=[0.2, 0.8], categories=["A", "B"])
        >>> dist.sample(3, random_state=42).equals(dist.sample(3, random_state=42))
        True
        """
        return self._sample(n_samples=n_samples, random_state=random_state)

    def _sample(self, n_samples=None, random_state=None):
        """Sample from the distribution.

        Each sampled value looks up one uniform draw in the cumulative probabilities of its row (inverse-cdf sampling).

        Parameters
        ----------
        n_samples : int, optional, default = None
        random_state : int, np.random.Generator, or None, optional, default = None

        Returns
        -------
        scalar or pd.DataFrame

        """
        rng = np.random.default_rng(random_state)
        n_draws = 1 if n_samples is None else n_samples
        n_rows, n_states = self._probs.shape

        # Dividing by the total makes the last cumulative value exactly 1, as in numpy's Generator.choice, so rounding
        # can never select a trailing zero-probability category.
        cdf = np.cumsum(self._probs, axis=1)
        cdf /= cdf[:, -1:]
        uniform = rng.random((n_draws, n_rows))
        state_idx = np.zeros((n_draws, n_rows), dtype=int)
        for j in range(n_states - 1):
            state_idx += uniform >= cdf[:, j]
        sampled = self._categories[state_idx]

        if self.ndim == 0:
            if n_samples is None:
                return sampled[0, 0]
            return pd.DataFrame(sampled).infer_objects()

        index = self.index
        if n_samples is not None:
            if not isinstance(index, pd.MultiIndex):
                index = pd.MultiIndex.from_arrays([index])
            index = pd.MultiIndex(
                levels=[pd.RangeIndex(n_samples), *index.levels],
                codes=[np.repeat(np.arange(n_samples), n_rows), *[np.tile(code, n_samples) for code in index.codes]],
                names=[None, *index.names],
            )

        return pd.DataFrame(sampled.reshape(-1, 1), index=index, columns=self.columns).infer_objects()

    def plot(
        self, fun: str | None = "pmf", ax: "Axes | np.ndarray | None" = None, **kwargs: Any
    ) -> "Axes | tuple[Figure, np.ndarray]":
        """Plot the nominal probability mass function.

        A scalar distribution produces one bar plot. An array distribution produces one bar plot per row. Each category
        gets one evenly spaced bar, in the order of ``categories`` and labelled with the category, and the height of
        each bar represents the corresponding probability.

        For an array distribution, each subplot is labeled using the corresponding entry in ``index``. The first entry
        in ``columns`` is used as the figure title.

        Parameters
        ----------
        fun : {"pmf"} or None, default="pmf"
            Distribution function to plot. ``None`` means ``"pmf"``. Currently, only the probability mass function
            (``"pmf"``) is supported.
        ax : matplotlib Axes object or array of Axes, optional
            Axes to plot in, one per row of the distribution. A scalar distribution defaults to the current axes
            (``plt.gca()``). An array distribution creates one subplot per row when axes are not provided.
        kwargs : keyword arguments
            ``sharex`` and ``sharey`` (default ``True``) are used when creating the subplots, as in skpro. All other
            keyword arguments are passed to ``Axes.bar``.

        Returns
        -------
        matplotlib.Axes or tuple of (matplotlib.Figure, np.ndarray)
            A scalar distribution returns its Axes. An array distribution returns the Figure and a one-dimensional array
            containing one Axes per row.

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
        >>> fig, axes = dist.plot(fun="pmf")  # doctest: +SKIP
        """
        _check_soft_dependencies("matplotlib", obj="distribution plot")
        import matplotlib.pyplot as plt

        if fun is None:
            fun = "pmf"
        if fun != "pmf":
            raise NotImplementedError("`NominalDistribution` only supports `pmf` currently")

        n_rows, n_states = self._probs.shape
        sharex = kwargs.pop("sharex", True)
        sharey = kwargs.pop("sharey", True)

        if self.ndim == 0 and ax is None:
            ax = plt.gca()

        if ax is None:
            fig, axes = plt.subplots(n_rows, 1, squeeze=False, sharex=sharex, sharey=sharey)
        else:
            axes = np.asarray(ax, dtype=object).reshape(-1, 1)
            if axes.shape[0] != n_rows:
                raise ValueError(
                    f"ax must contain one Axes per row of the distribution: expected {n_rows}, got {axes.shape[0]}"
                )
            fig = axes[0, 0].figure

        positions = np.arange(n_states)
        labels = [str(category) for category in self._categories]
        for i in range(n_rows):
            current_ax = axes[i, 0]
            current_ax.bar(positions, self._probs[i], **kwargs)
            current_ax.set_xticks(positions, labels)
            current_ax.set_ylabel("probability" if self.ndim == 0 else str(self.index[i]))
            current_ax.set_ylim(0, 1)

        if self.ndim > 0:
            axes[0, 0].set_title(str(self.columns[0]))
        axes[-1, 0].set_xlabel("categories")

        if self.ndim == 0:
            return axes[0, 0]
        return fig, axes[:, 0]

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns.

        Returns
        -------
        dict

        """
        # There is only one column, so nothing to subset, but an out-of-range position must raise IndexError.
        if colidx is not None:
            self.columns[colidx]

        probs = self._probs
        if rowidx is not None:
            probs = probs[rowidx, :]
            if probs.ndim == 1 and not coerce_scalar:
                probs = probs.reshape(1, -1)

        return {"probs": probs, "categories": self._categories}

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> list[dict[str, Any]]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no special parameters are defined for a
            value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class. Each dict are parameters to construct an "interesting"
            test instance, i.e., `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        params1 = {"probs": [[0.1, 0.9], [0.7, 0.3]], "categories": [1, 2]}
        params2 = {"probs": [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], "categories": [1, 2, 3]}
        params3 = {"probs": [0.2, 0.8], "categories": ["A", "B"]}
        return [params1, params2, params3]
