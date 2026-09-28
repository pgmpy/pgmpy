import inspect
from collections.abc import Hashable
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.api.types import is_complex_dtype, is_float_dtype, is_numeric_dtype
from skbase.base import BaseEstimator
from skbase.utils.dependencies import _check_estimator_deps
from sklearn.utils import Tags, TargetTags
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from pgmpy.parameterization.distributions.nominal import _sample_index
from pgmpy.utils import get_dataset_type


class BaseParameter(BaseEstimator):
    """Base class for parameterizations: models of one variable given its parent variables.

    ``fit(X, y)`` learns the distribution of the target ``y`` given the parents ``X``, ``predict_proba(X)`` returns that
    distribution for each row of ``X``, and ``sample`` draws values of the target. A root variable has no parents: fit
    it with ``X=None`` or with a DataFrame without columns. Parents are sorted by name, so the order of ``X``'s columns
    doesn't matter.

    Subclasses implement ``_fit``, which gets ``X`` with its columns in ``evidence_`` order, and ``_predict_proba``,
    where ``_predict_proba(None)`` returns the marginal distribution of the target. They set the ``variable_type`` and
    ``parent_type`` tags to the types they support for the target and for the parents (``"discrete"``, ``"continuous"``
    or both); continuous-only parents must be numeric, while discrete ones can have any labels. They also set
    ``supports_weighted_data`` if ``_fit`` uses sample weights, and ``python_dependencies`` to the packages checked at
    construction. The default ``_sample`` passes ``random_state`` to the ``sample`` method of the predicted
    distribution. If that method doesn't take it, as for skpro's distributions, it samples as skpro does, by applying
    the distribution's ``ppf`` to uniform values, drawn from ``numpy.random.default_rng(random_state)``.

    Attributes
    ----------
    variable_ : hashable
        Name of the target variable.
    evidence_ : list
        Names of the parent variables, sorted; empty for a root variable. Names of different types are sorted by the
        name of their type first.
    variable_type_ : str
        Type of the target, ``"discrete"`` or ``"continuous"``.

    Warnings
    --------
    Experimental: the API of ``pgmpy.parameterization`` may change in any release without a deprecation period.
    """

    _tags = {
        "object_type": "parameterization",
        "variable_type": ["discrete", "continuous"],
        "parent_type": ["discrete", "continuous"],
        "supports_weighted_data": False,
        "python_dependencies": None,
    }

    def __init__(self) -> None:
        _check_estimator_deps(self)
        super().__init__()

    def fit(
        self, X: pd.DataFrame | None, y: pd.DataFrame | pd.Series, sample_weight: ArrayLike | None = None
    ) -> "BaseParameter":
        """Fit the distribution of ``y`` given ``X``.

        Parameters
        ----------
        X : pandas.DataFrame or None
            Values of the parent variables, one column per parent. ``None`` or a DataFrame without columns for a root
            variable.
        y : pandas.DataFrame or pandas.Series
            Values of the target variable, as a single column with the same index as ``X``.
        sample_weight : array-like of shape (n_samples,), optional
            Weight of each row: finite, non-negative and not all zero. Only for classes with the
            ``supports_weighted_data`` tag.

        Returns
        -------
        self
            The fitted instance.
        """
        self.reset()

        if isinstance(y, pd.Series):
            y = y.to_frame()
        if not isinstance(y, pd.DataFrame) or y.shape[1] != 1:
            raise ValueError("y must be a pandas Series or a DataFrame with exactly one column.")
        if X is None:
            X = pd.DataFrame(index=y.index)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame, or None for a root variable.")
        if X.columns.has_duplicates:
            raise ValueError(f"X has repeated column names: {X.columns[X.columns.duplicated()].unique().tolist()}.")
        check_consistent_length(X, y)
        if len(y) == 0:
            raise ValueError("fit needs at least one row of data.")
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index.")
        if y.columns[0] in X.columns:
            raise ValueError(f"The target {y.columns[0]!r} must not also be a column of X.")
        _check_values(X, "X")
        _check_values(y, "y")

        supported_types = _as_list(self.get_tag("variable_type"))
        variable_type = supported_types[0] if len(supported_types) == 1 else get_dataset_type(y)
        # For one column, get_dataset_type says "mixed" only for an ordered categorical with non-string categories.
        if variable_type == "mixed":
            variable_type = "discrete"
        if variable_type not in supported_types:
            raise ValueError(f"{type(self).__name__} supports {supported_types} targets, but y is {variable_type}.")
        if variable_type == "continuous" and not is_numeric_dtype(y.iloc[:, 0]):
            raise ValueError(f"{type(self).__name__} needs a numeric target, but y has dtype {y.dtypes.iloc[0]}.")
        self._check_parent_types(X)

        if sample_weight is not None:
            if not self.get_tag("supports_weighted_data"):
                raise ValueError(f"{type(self).__name__} does not support sample_weight.")
            if isinstance(sample_weight, pd.Series) and not sample_weight.index.equals(y.index):
                raise ValueError("A sample_weight Series must have the same index as y.")
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (len(y),):
                raise ValueError(
                    f"sample_weight needs one value per row, shape ({len(y)},), but has shape {sample_weight.shape}."
                )
            if not np.isfinite(sample_weight).all() or (sample_weight < 0).any() or not sample_weight.any():
                raise ValueError("sample_weight must be finite and non-negative, with at least one positive value.")

        self.variable_ = y.columns[0]
        self.evidence_ = [X.columns[position] for position in _parent_order(list(X.columns))]
        self.variable_type_ = variable_type
        self._fit(X[self.evidence_], y, sample_weight)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | None = None) -> Any:
        """Return the distribution of the target for each row of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame, optional
            Values of the parent variables, with the columns seen in ``fit`` in any order. For a root variable, a
            DataFrame without columns, whose index sets the rows. ``None`` for the marginal distribution of the target
            over the parent values seen in ``fit``.

        Returns
        -------
        skpro distribution
            The distribution of the target, with one row per row of ``X`` and the same index. Without ``X``, its
            marginal distribution.
        """
        return self._predict_proba(self._check_X(X))

    def sample(
        self,
        X: pd.DataFrame | None = None,
        n_samples: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Any:
        """Draw values of the target variable.

        Parameters
        ----------
        X : pandas.DataFrame, optional
            Values of the parent variables, as for ``predict_proba``. Without ``X``, values are drawn from the marginal
            distribution of the target.
        n_samples : int, optional
            Number of values to draw for each row of ``X``, one if omitted. Without ``X``, the total number of values.
        random_state : int, numpy.random.Generator or None, optional
            Controls the randomness of this call. An ``int`` gives the same values on every call, a ``Generator`` is
            used as-is and advances across calls, and ``None`` draws fresh values.

        Returns
        -------
        pandas.DataFrame
            One column named after the target. With ``X``, it has ``X``'s index, and with ``n_samples`` also an unnamed
            first index level numbering the draws.
        """
        if n_samples is not None and (
            isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 0
        ):
            raise ValueError(f"n_samples must be a non-negative integer, but got {n_samples!r}.")
        X = self._check_X(X)
        if X is None and n_samples is None:
            raise ValueError("Pass X, or n_samples to draw from the marginal distribution.")
        return self._sample(X, n_samples, random_state)

    def _check_X(self, X: pd.DataFrame | None) -> pd.DataFrame | None:
        check_is_fitted(self)
        if X is None:
            return None
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if len(X.columns) != len(self.evidence_) or set(X.columns) != set(self.evidence_):
            raise ValueError(f"X must have the columns seen in fit, {self.evidence_}, but has {list(X.columns)}.")
        _check_values(X, "X")
        self._check_parent_types(X)
        return X[self.evidence_]

    def _check_parent_types(self, X: pd.DataFrame) -> None:
        if "discrete" in _as_list(self.get_tag("parent_type")):
            return
        non_numeric = {column: str(dtype) for column, dtype in X.dtypes.items() if not is_numeric_dtype(dtype)}
        if non_numeric:
            raise ValueError(f"{type(self).__name__} needs numeric parents, but X has other columns: {non_numeric}.")

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        raise NotImplementedError

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        raise NotImplementedError

    def _sample(
        self, X: pd.DataFrame | None, n_samples: int | None, random_state: int | np.random.Generator | None
    ) -> pd.DataFrame:
        distribution = self._predict_proba(X)
        if "random_state" in inspect.signature(distribution.sample).parameters:
            samples = distribution.sample(n_samples, random_state=random_state)
        else:
            samples = _sample_from_ppf(distribution, n_samples, np.random.default_rng(random_state))
        # Without X the distribution is scalar, and its samples come in a column named 0.
        return samples.set_axis([self.variable_], axis=1)

    def __sklearn_tags__(self) -> Tags:
        return Tags(estimator_type=None, target_tags=TargetTags(required=True))

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or self.is_fitted != other.is_fitted:
            return False
        if not self.is_fitted:
            return super().__eq__(other)

        fitted, other_fitted = self.get_fitted_params(deep=False), other.get_fitted_params(deep=False)
        if fitted.keys() != other_fitted.keys():
            return False
        for name, value in fitted.items():
            other_value = other_fitted[name]
            if isinstance(value, (np.ndarray, float, np.floating)):
                if np.shape(value) != np.shape(other_value) or not np.allclose(value, other_value):
                    return False
            elif value != other_value:
                return False
        return True

    def __hash__(self) -> int:
        if not self.is_fitted:
            return hash(type(self).__name__)
        return hash((type(self).__name__, self.variable_, tuple(self.evidence_)))


def _as_list(types: str | list[str]) -> list[str]:
    """Return the value of a type tag, a type or a list of types, as a list."""
    return [types] if isinstance(types, str) else types


def _check_values(data: pd.DataFrame, name: str) -> None:
    """Raise a ValueError if ``data`` has missing, complex or infinite values."""
    if data.isna().to_numpy().any():
        raise ValueError(f"{name} must not contain missing values.")
    if any(is_complex_dtype(dtype) for dtype in data.dtypes):
        raise ValueError(f"{name} must not contain complex values.")
    floats = data.loc[:, [is_float_dtype(dtype) for dtype in data.dtypes]]
    if np.isinf(floats.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} must not contain infinite values.")


def _check_names(variable: Hashable, evidence: list) -> None:
    """Raise a TypeError if a name isn't hashable, and a ValueError if the variable and its parents share a name."""
    if len({variable, *evidence}) != 1 + len(evidence):
        raise ValueError(
            f"The variable and its parents must have different names, but got {variable!r} and {evidence}."
        )


def _parent_order(evidence: list) -> list[int]:
    """Return the positions of the parents in sorted order, sorting names of different types by type name first."""
    return sorted(range(len(evidence)), key=lambda position: (type(evidence[position]).__name__, evidence[position]))


def _sample_from_ppf(distribution: Any, n_samples: int | None, rng: np.random.Generator) -> pd.DataFrame:
    """Sample as skpro's default ``sample`` does, by applying the ``ppf`` to uniform values, but drawn from ``rng``."""
    if distribution.ndim == 0:
        return pd.DataFrame([distribution.ppf(p) for p in rng.random(n_samples)], columns=[0], dtype=float)
    uniform = rng.random((1 if n_samples is None else n_samples, *distribution.shape))
    values = [distribution.ppf(pd.DataFrame(u, distribution.index, distribution.columns)).to_numpy() for u in uniform]
    index = distribution.index if n_samples is None else _sample_index(distribution.index, n_samples)
    return pd.DataFrame(np.reshape(values, (-1, distribution.shape[1])), index=index, columns=distribution.columns)
