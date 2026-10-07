import inspect
from collections.abc import Hashable
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.base import BaseEstimator, BaseObject
from skbase.utils.dependencies import _check_estimator_deps
from sklearn.base import BaseEstimator as SklearnEstimator

from pgmpy.parameterization.distributions.nominal import _sample_index
from pgmpy.utils import preprocess_data


class BaseParameter(BaseEstimator):
    """Base class for parameterizations: models of one variable given its parents.

    ``fit(X, y)`` learns the distribution of the target ``y`` given the parents ``X``, ``predict_proba(X)`` returns it
    for each row of ``X``, ``predict(X)`` a point prediction from it, ``log_likelihood(X, y)`` scores ``y`` under it,
    and ``sample`` draws from it. Parents are sorted by name, so the order of ``X``'s columns doesn't matter. ``X=None``
    stands for no parents: it fits a root, and gives a root's own distribution. A variable with parents needs ``X``, as
    its marginal distribution depends on the rest of the network.

    Subclasses implement ``_fit``, which gets ``X`` in ``evidence_`` order, and ``_predict_proba``, which returns a
    root's own distribution for ``X=None``. Both select ``X``'s columns by position, as selecting them by label fails
    for some names, e.g. booleans. ``_predict_proba`` gets ``X`` with its own labels only from ``predict_proba``;
    ``predict``, ``log_likelihood`` and ``sample`` pass it on a RangeIndex. Subclasses also set a ``name`` and the other
    tags registered for parameterizations in :mod:`pgmpy.registry`; ``variable_type`` must hold a single type by the
    time ``fit`` runs.

    Attributes
    ----------
    variable_ : hashable
        Name of the target.
    evidence_ : list
        Names of the parents, sorted; empty for a root.
    variable_type_ : str
        Type of the target, ``"discrete"`` or ``"continuous"``.
    """

    _tags = {
        "object_type": "parameterization",
        "name": None,
        "variable_type": ["discrete", "continuous"],
        "parent_data_types": ["discrete", "continuous", "mixed"],
        "supports_weighted_data": False,
        "requires_data": True,
        "capability:factor": False,
        "capability:exact_inference": False,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
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
            Parent values, one column per parent; ``None`` for a root. Continuous-only parents must be numeric.
        y : pandas.Series or pandas.DataFrame
            Target values, in a single column with ``X``'s index. A continuous target must be numeric.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative frequency of each row: weight 2 counts a row twice. For classes with the
            ``supports_weighted_data`` tag.

        Returns
        -------
        self
            The fitted instance.
        """
        # Step 1: Reset any earlier fit, and check that X and y are DataFrames with the same index and no missing
        # values, with at least one row if the class needs data.
        self.reset()
        y = _as_frame(y)
        if X is None:
            X = pd.DataFrame(index=y.index)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame, or None for a root variable.")
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index.")
        for data, name in ((X, "X"), (y, "y")):
            if data.isna().to_numpy().any():
                raise ValueError(f"{name} must not contain missing values.")
        if len(y) == 0 and self.get_tag("requires_data"):
            raise ValueError(f"{type(self).__name__} needs at least one row of data.")

        # Step 2: Check that a continuous target and continuous-only parents are numeric.
        [variable_type] = self.get_tag("variable_type")
        if variable_type == "continuous" and preprocess_data(y)[1][y.columns[0]] != "N":
            raise ValueError(f"{type(self).__name__} needs a numeric target, but y has dtype {y.dtypes.iloc[0]}.")
        if self.get_tag("parent_data_types") == ["continuous"]:
            non_numeric = [column for column, kind in preprocess_data(X)[1].items() if kind != "N"]
            if non_numeric:
                raise ValueError(f"{type(self).__name__} needs numeric parents, but {non_numeric} aren't numeric.")

        # Step 3: Check the sample weights, if any.
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

        # Step 4: Check the parents' names as from_values does, sort the parents by name and fit the subclass's model on
        # X in that order. Columns are selected by position, as X[names] would read a list of boolean names, e.g.
        # [False, True], as a row mask.
        columns = _checked_evidence(y.columns[0], list(X.columns))
        order = _parent_order(columns)
        self.variable_ = y.columns[0]
        self.evidence_ = [columns[position] for position in order]
        self.variable_type_ = variable_type
        self._fit(X.iloc[:, order], y, sample_weight)
        self._is_fitted = True
        return self

    def set_params(self, **params: Any) -> "BaseParameter":
        """Set the parameters, as skbase does, and reset again so that tags taken from them, also from nested ones such
        as ``estimator__alpha``, follow the new values."""
        # skbase resets before it sets nested parameters, so the tags set in __init__ would follow the old ones.
        return super().set_params(**params).reset()

    def predict_proba(self, X: pd.DataFrame | None = None) -> Any:
        """Return the distribution of the target for each row of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame, optional
            Parent values, with the columns seen in ``fit`` in any order. For a root, a DataFrame without columns, or
            ``None`` for the root's own distribution.

        Returns
        -------
        skpro distribution
            One row per row of ``X``, with ``X``'s index; a scalar distribution for a root without ``X``.
        """
        return self._predict_proba(self._check_X(X))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a point prediction of the target for each row of ``X``.

        The mean of the distribution from ``predict_proba`` for a continuous target, and its most probable state for a
        discrete one, the first in category order on ties.

        Parameters
        ----------
        X : pandas.DataFrame
            Parent values, as for ``predict_proba``; for a root, a DataFrame without columns.

        Returns
        -------
        pandas.DataFrame
            One column named after the target, with ``X``'s index.
        """
        X = self._check_X(X)
        if X is None:
            raise ValueError("predict needs X; for a root variable, pass a DataFrame without columns.")
        return self._predict(X)

    def log_likelihood(self, X: pd.DataFrame | None, y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Return the log-likelihood of each row of ``y`` given the same row of ``X``, log p(y | x).

        The log of the density for a distribution over a continuous measure, and of the probability for one over a
        discrete measure, such as a ``NominalDistribution``.

        Parameters
        ----------
        X : pandas.DataFrame or None
            Parent values, as for ``predict``; ``None`` for a root.
        y : pandas.Series or pandas.DataFrame
            Target values, in a single column with ``X``'s index.

        Returns
        -------
        pandas.DataFrame
            One column named after the target, with ``X``'s index.
        """
        # Step 1: Check X and y as predict and fit do.
        y = _as_frame(y)
        X = self._check_X(X)
        if X is None:
            X = pd.DataFrame(index=y.index)
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index.")

        # Step 2: Predict the distributions on a RangeIndex and put X's index back at the end, as skpro matches values
        # to rows by label, which mixes up rows with repeated labels.
        distribution = self._predict_proba(X.set_axis(pd.RangeIndex(len(X))))

        # Step 3: Score y with the log-probability under a discrete measure, and the log-density under any other. y goes
        # in as a numpy array, which skpro matches to rows by position.
        values = y.to_numpy()
        discrete = distribution.get_tag("distr:measuretype", "continuous", raise_error=False) == "discrete"
        scores = distribution.log_pmf(values) if discrete else distribution.log_pdf(values)
        return pd.DataFrame(scores.to_numpy(), index=X.index, columns=[self.variable_])

    def sample(
        self,
        X: pd.DataFrame | None = None,
        n_samples: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Any:
        """Draw values of the target from its distribution given each row of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame, optional
            Parent values, as for ``predict_proba``; ``None`` draws from a root's own distribution.
        n_samples : int, optional
            Number of draws per row of ``X``, one if omitted; without ``X``, the total number of draws.
        random_state : int, numpy.random.Generator or None, optional
            Seed or generator for this call; ``None`` draws fresh values.

        Returns
        -------
        pandas.DataFrame
            One column named after the target, with ``X``'s index, preceded by a level numbering the draws when
            ``n_samples`` is given.
        """
        if n_samples is not None and (
            isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 0
        ):
            raise ValueError(f"n_samples must be a non-negative integer, but got {n_samples!r}.")
        X = self._check_X(X)
        if X is None and n_samples is None:
            raise ValueError("Pass X, or n_samples to draw that many values from the root's distribution.")
        return self._sample(X, n_samples, random_state)

    def _check_X(self, X: pd.DataFrame | None) -> pd.DataFrame | None:
        self.check_is_fitted()
        if X is None:
            if self.evidence_:
                raise ValueError(
                    f"{self.variable_!r} has parents {self.evidence_}, so pass their values in X; X=None is only for a "
                    "root. Its marginal distribution depends on its parents' distributions, so get it from the network."
                )
            return None
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if len(X.columns) != len(self.evidence_) or set(X.columns) != set(self.evidence_):
            raise ValueError(f"X must have the columns seen in fit, {self.evidence_}, but has {list(X.columns)}.")
        return X.iloc[:, [X.columns.get_loc(name) for name in self.evidence_]]

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        raise NotImplementedError

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        raise NotImplementedError

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Predict the distribution of the target for each row, on a RangeIndex, as skpro matches rows by label.
        distribution = self._predict_proba(X.set_axis(pd.RangeIndex(len(X))))

        # Step 2: Reduce it to a point: the mean for a continuous target, and the most probable state for a discrete
        # one. A Series keeps tuple labels whole, and gives the states' dtype as NominalDistribution.sample does.
        if self.variable_type_ == "continuous":
            return pd.DataFrame(distribution.mean().to_numpy(), index=X.index, columns=[self.variable_])
        categories = pd.Series(distribution.categories)
        values = categories.to_numpy()[np.argmax(distribution.probs, axis=1)]
        dtype = categories.infer_objects().dtype
        return pd.DataFrame(values.reshape(-1, 1), index=X.index, columns=[self.variable_]).astype(dtype)

    def _sample(
        self, X: pd.DataFrame | None, n_samples: int | None, random_state: int | np.random.Generator | None
    ) -> pd.DataFrame:
        # Step 1: Repeat X's rows once per draw, or for a root without X, take n_samples rows without columns, and
        # predict their distributions on a RangeIndex: skpro matches values to rows by label, which mixes up rows with
        # repeated labels.
        if X is None:
            rows = pd.DataFrame(index=pd.RangeIndex(n_samples))
        else:
            rows = X.iloc[np.tile(np.arange(len(X)), 1 if n_samples is None else n_samples)]
        distribution = self._predict_proba(rows.set_axis(pd.RangeIndex(len(rows))))

        # Step 2: Draw one value per row, with the distribution's own sample() if it takes a random_state, as
        # NominalDistribution's does, and otherwise, as for skpro's distributions, with the ppf of uniform values from
        # the seeded generator.
        if "random_state" in inspect.signature(distribution.sample).parameters:
            samples = distribution.sample(random_state=random_state)
        else:
            uniform = np.random.default_rng(random_state).random((len(rows), 1))
            samples = pd.DataFrame(distribution.ppf(uniform).to_numpy(), dtype=float)

        # Step 3: Label the samples with X's index, and a level numbering the draws, and with the target's name.
        if X is not None:
            samples = samples.set_axis(X.index if n_samples is None else _sample_index(X.index, n_samples))
        return samples.set_axis([self.variable_], axis=1)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or self.is_fitted != other.is_fitted:
            return False
        if not self.is_fitted:
            return _equal(self.get_params(deep=False), other.get_params(deep=False), fitted=False)
        if (self.variable_, self.evidence_) != (other.variable_, other.evidence_):
            return False
        return _equal(self.get_fitted_params(deep=False), other.get_fitted_params(deep=False))

    def __hash__(self) -> int:
        if not self.is_fitted:
            return hash(type(self).__name__)
        return hash((type(self).__name__, self.variable_, tuple(self.evidence_)))


def _equal(value: Any, other: Any, fitted: bool = True) -> bool:
    """Return whether two values are equal: numbers and numeric arrays within numpy.allclose's tolerance, containers
    item by item, and skbase or sklearn objects by their parameters, except fitted estimators, which compare by
    identity."""
    if all(
        isinstance(v, (Real, np.number, np.ndarray)) and np.issubdtype(np.asarray(v).dtype, np.number)
        for v in (value, other)
    ):
        return np.shape(value) == np.shape(other) and bool(np.allclose(value, other))
    if type(value) is not type(other):
        return False
    if isinstance(value, np.ndarray):
        return np.array_equal(value, other)
    if isinstance(value, (pd.Index, pd.Series, pd.DataFrame, pd.api.extensions.ExtensionArray)):
        return value.equals(other)
    if isinstance(value, dict):
        return value.keys() == other.keys() and all(_equal(value[key], other[key], fitted) for key in value)
    if isinstance(value, (list, tuple)):
        return len(value) == len(other) and all(_equal(a, b, fitted) for a, b in zip(value, other))
    # Fitted estimators hold what they learned in attributes that skbase and sklearn don't compare.
    if fitted and isinstance(value, (BaseEstimator, SklearnEstimator)):
        return value is other
    if isinstance(value, (BaseObject, SklearnEstimator)):
        return _equal(value.get_params(deep=False), other.get_params(deep=False), fitted)
    return value == other


def _as_frame(y: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return the target values ``y`` as a DataFrame, checking that they are a Series or a one-column DataFrame."""
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if not isinstance(y, pd.DataFrame) or y.shape[1] != 1:
        raise ValueError("y must be a pandas Series or a DataFrame with exactly one column.")
    return y


def _checked_evidence(variable: Hashable, evidence: list | tuple | None) -> list:
    """Return ``evidence`` as a list, checking that the variable and its parents have distinct names."""
    evidence = [] if evidence is None else evidence
    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a list or tuple of parent names, but is a {type(evidence).__name__}.")
    evidence = list(evidence)
    if len({variable, *evidence}) != 1 + len(evidence):
        raise ValueError(
            f"The variable and its parents must have different names, but got {variable!r} and {evidence}."
        )
    return evidence


def _name_key(name: Hashable) -> tuple:
    """Return a sort key for any name: numbers by value, then strings, then tuples element by element, then other names
    by type and repr."""
    if isinstance(name, Real):
        return 0, name
    if isinstance(name, str):
        return 1, name
    if isinstance(name, tuple):
        return 2, tuple(_name_key(part) for part in name)
    return 3, type(name).__name__, repr(name)


def _parent_order(evidence: list) -> list[int]:
    """Return the positions of ``evidence`` in sorted order."""
    return sorted(range(len(evidence)), key=lambda position: _name_key(evidence[position]))
