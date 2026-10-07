from typing import Any, NoReturn

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

BaseDistribution = _safe_import("skpro.distributions.base.BaseDistribution")


class NominalDistribution(BaseDistribution):
    """Nominal distribution for discrete random variables.

    Represents one or more nominal categorical probability distributions over a finite set of discrete states. A
    one-dimensional ``probs`` vector without ``index`` or ``columns`` defines a scalar distribution with shape ``()``. A
    two-dimensional ``probs`` array defines a distribution of shape ``(n_instances, 1)``, with each row assigning
    probability mass to the states specified by ``categories``.

    The categories are treated as *nominal* (unordered, non-numeric) labels. Consequently, order-, arithmetic- and
    density-based summaries are undefined and raise ``NotImplementedError``: ``cdf``, ``ppf``, ``surv``, ``haz``,
    ``mean``, ``var``, ``energy`` and ``pdfnorm``. The supported methods are ``pmf``, ``log_pmf`` and ``sample``.

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
    >>> from pgmpy.parameterization import NominalDistribution
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

        # Step 1: Check that probs has one or two dimensions, and holds non-negative probabilities with every row
        # summing to 1, with the same tolerance as TabularCPD.
        probs_arr = np.asarray(probs, dtype=float)
        if probs_arr.ndim not in (1, 2):
            raise ValueError("probs must be a one- or two-dimensional array")
        is_scalar = probs_arr.ndim == 1 and index is None and columns is None
        probs_arr = np.atleast_2d(probs_arr)
        if np.any(probs_arr < 0):
            raise ValueError("probs must contain only non-negative probabilities")
        row_sums = probs_arr.sum(axis=1)
        invalid_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=0.01))
        if invalid_rows.size:
            raise ValueError(
                f"The probabilities in each row of probs must sum to 1; invalid row indices: {invalid_rows.tolist()}, "
                f"row sums: {row_sums[invalid_rows].tolist()}"
            )

        # Step 2: Check the categories, keeping each label as given, including tuples and mixed types, in an object
        # array.
        if isinstance(categories, str):
            raise TypeError(f"categories must be an array-like of labels, not a single string: {categories!r}")
        categories_arr = np.fromiter(categories, dtype=object)
        if pd.isna(categories_arr).any():
            raise ValueError(f"Categories must not contain missing values: {categories}")

        # Step 3: Check that the categories are unique, with one per column of probs.
        if len(categories_arr) != len(set(categories_arr)):
            raise ValueError(f"Categories must contain unique values: {categories}")

        if probs_arr.shape[1] != len(categories_arr):
            raise ValueError(
                f"mismatch between the shape of categories and probs : {len(categories_arr)}, {probs_arr.shape[1]}"
            )

        # Step 4: Check the index and columns of an array distribution, or give them defaults.
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

        # Step 5: Store the probabilities, normalized to sum exactly to 1, and the categories.
        self._probs = probs_arr / row_sums[:, np.newaxis]
        self._categories = categories_arr

        super().__init__(index=index, columns=columns)

    def _select_probs(self, x):
        """Return whether each queried value is a category, and its probability, which is arbitrary where it isn't."""
        x = np.atleast_2d(x)
        if pd.isna(x).any():
            raise ValueError("Queries must not contain missing values (None, NaN or pd.NA)")

        matches = x == self._categories
        valid = matches.any(axis=1)
        state_idx = matches.argmax(axis=1)
        row_idx = np.arange(self._probs.shape[0])

        return valid, self._probs[row_idx, state_idx]

    def _pmf(self, x):
        """Return the probability of each queried value, 0 for a value that isn't a category."""
        valid, selected = self._select_probs(x)
        res = np.where(valid, selected, 0.0)
        return res.reshape(self.shape)

    def _log_pmf(self, x):
        """Return the log-probability of each queried value, ``-inf`` for a value that isn't a category."""
        valid, selected = self._select_probs(x)
        with np.errstate(divide="ignore"):
            log_selected = np.log(selected)
        res = np.where(valid, log_selected, -np.inf)
        return res.reshape(self.shape)

    def cdf(self, x: ArrayLike) -> NoReturn:
        """Not defined: categories have no inherent order."""
        raise NotImplementedError("cdf is not defined for NominalDistribution: categories have no inherent order.")

    def ppf(self, p: ArrayLike) -> NoReturn:
        """Not defined: categories have no inherent order."""
        raise NotImplementedError("ppf is not defined for NominalDistribution: categories have no inherent order.")

    def surv(self, x: ArrayLike) -> NoReturn:
        """Not defined: categories have no inherent order."""
        raise NotImplementedError("surv is not defined for NominalDistribution: categories have no inherent order.")

    def haz(self, x: ArrayLike) -> NoReturn:
        """Not defined: categories have no inherent order."""
        raise NotImplementedError("haz is not defined for NominalDistribution: categories have no inherent order.")

    def mean(self) -> NoReturn:
        """Not defined: categories are not numeric."""
        raise NotImplementedError("mean is not defined for NominalDistribution: categories are not numeric.")

    def var(self) -> NoReturn:
        """Not defined: categories are not numeric."""
        raise NotImplementedError("var is not defined for NominalDistribution: categories are not numeric.")

    def energy(self, x: ArrayLike | None = None) -> NoReturn:
        """Not defined: categories have no metric."""
        raise NotImplementedError("energy is not defined for NominalDistribution: categories have no metric.")

    def pdfnorm(self, a: float = 2) -> NoReturn:
        """Not defined: there is no probability density."""
        raise NotImplementedError("pdfnorm is not defined for NominalDistribution: it has no probability density.")

    def sample(self, n_samples: int | None = None, random_state: int | np.random.Generator | None = None) -> Any:
        """Draw categories from the distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of draws, one if omitted.
        random_state : int, numpy.random.Generator or None, optional
            Seed or generator for this call; ``None`` draws fresh values.

        Returns
        -------
        scalar or pandas.DataFrame
            For a scalar distribution, one category, or with ``n_samples`` a DataFrame with one row per draw. For an
            array distribution, a DataFrame with its index and columns, preceded by a level numbering the draws when
            ``n_samples`` is given.

        Examples
        --------
        >>> dist = NominalDistribution(probs=[0.2, 0.8], categories=["A", "B"])
        >>> dist.sample(3, random_state=42).equals(dist.sample(3, random_state=42))
        True
        """
        return self._sample(n_samples=n_samples, random_state=random_state)

    def _sample(self, n_samples=None, random_state=None):
        # Step 1: Compute the cumulative probabilities of each row. Dividing by the total makes the last one exactly 1,
        # as in numpy's Generator.choice, so rounding can never select a trailing zero-probability category.
        rng = np.random.default_rng(random_state)
        n_draws = 1 if n_samples is None else n_samples
        n_rows, n_states = self._probs.shape
        cdf = np.cumsum(self._probs, axis=1)
        cdf /= cdf[:, -1:]

        # Step 2: Draw one uniform value per draw and row, and look it up in the cumulative probabilities of its row.
        uniform = rng.random((n_draws, n_rows))
        state_idx = np.zeros((n_draws, n_rows), dtype=int)
        for j in range(n_states - 1):
            state_idx += uniform >= cdf[:, j]
        sampled = self._categories[state_idx]

        # Step 3: Return the categories drawn, with the dtype of the categories rather than of the values drawn, so that
        # it is the same for any number of draws: one category for a single draw from a scalar distribution, and a
        # DataFrame otherwise.
        dtype = pd.Series(self._categories).infer_objects().dtype

        if self.ndim == 0:
            if n_samples is None:
                return sampled[0, 0]
            return pd.DataFrame(sampled).astype(dtype)

        index = self.index if n_samples is None else _sample_index(self.index, n_samples)
        return pd.DataFrame(sampled.reshape(-1, 1), index=index, columns=self.columns).astype(dtype)

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
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
        """Return the parameters of the instances that skbase's and skpro's estimator checks test."""
        params1 = {"probs": [[0.1, 0.9], [0.7, 0.3]], "categories": [1, 2]}
        params2 = {"probs": [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], "categories": [1, 2, 3]}
        params3 = {"probs": [0.2, 0.8], "categories": ["A", "B"]}
        return [params1, params2, params3]


def _sample_index(index: pd.Index, n_samples: int) -> pd.MultiIndex:
    """Return the index of ``n_samples`` draws for each row of ``index``, with a first level numbering the draws."""
    if not isinstance(index, pd.MultiIndex):
        index = pd.MultiIndex.from_arrays([index])
    return pd.MultiIndex(
        levels=[pd.RangeIndex(n_samples), *index.levels],
        codes=[np.repeat(np.arange(n_samples), len(index)), *[np.tile(code, n_samples) for code in index.codes]],
        names=[None, *index.names],
    )
