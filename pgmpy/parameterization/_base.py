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


class BaseParameterization(BaseEstimator):
    """Base class for parameterizations: models of one variable given its parents.

    ``fit(X, y)`` learns the distribution of the target ``y`` given the parents ``X``, ``predict_proba(X)`` returns it
    for each row of ``X``, ``predict(X)`` a point prediction from it, ``log_likelihood(X, y)`` scores ``y`` under it,
    and ``sample`` draws from it. Parents are sorted by name, so the order of ``X``'s columns doesn't matter. ``X=None``
    stands for no parents: it fits a root, and gives a root's own distribution. A variable with parents needs ``X``, as
    its marginal distribution depends on the rest of the network.

    Subclasses implement ``_fit``, which gets ``X`` in ``evidence_`` order, and ``_predict_proba``, which returns a
    root's own distribution for ``X=None``, and for a discrete target a ``NominalDistribution``. Both select ``X``'s
    columns by position, as pandas reads a list of boolean names, e.g. ``X[[False, True]]``, as a row mask.
    ``_predict_proba`` gets ``X`` with its own labels only from ``predict_proba``; ``predict``, ``log_likelihood`` and
    ``sample`` pass it on a RangeIndex. Subclasses also set a ``name`` and the other tags registered for
    parameterizations in :mod:`pgmpy.registry`; ``variable_type`` must hold a single type by the time ``fit`` runs.

    Unfitted objects compare their class and parameters. A fitted object equals only itself, unless its class compares
    what it learned in ``_fitted_equal``, as the CPDs and ``DistributionAdapter`` do. The hash is constant per class.

    Attributes
    ----------
    variable_ : hashable
        Name of the target.
    evidence_ : list
        Names of the parents, sorted; empty for a root.
    """

    _tags = {
        "object_type": "parameterization",
        "name": None,
        "variable_type": ["discrete", "continuous"],
        "parent_data_types": ["discrete", "continuous", "mixed"],
        "supports_weighted_data": False,
        "requires_data": True,
        "capability:distribution": True,
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
    ) -> "BaseParameterization":
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
        # Step 1: Reset any earlier fit, and check that X and y are DataFrames with the same index, distinct names, as
        # from_values requires, and no missing values, with at least one row if the class needs data.
        self.reset()
        y = _as_frame(y)
        if X is None:
            X = pd.DataFrame(index=y.index)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame, or None for a root variable.")
        columns = _checked_evidence(y.columns[0], list(X.columns))
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

        # Step 4: Sort the parents by name and fit the subclass's model on X in that order. Columns are selected by
        # position, as X[names] would read a list of boolean names, e.g. [False, True], as a row mask.
        order = _parent_order(columns)
        self.variable_ = y.columns[0]
        self.evidence_ = [columns[position] for position in order]
        self._fit(X.iloc[:, order], y, sample_weight)
        self._is_fitted = True
        return self

    def set_params(self, **params: Any) -> "BaseParameterization":
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
        return self._predict(self._check_X(X, "predict"))

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
        y = _as_frame(y)
        X = self._check_X(X)
        if X is None:
            X = pd.DataFrame(index=y.index)
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index.")
        return self._log_likelihood(X, y)

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
        if n_samples is not None:
            _check_integer("n_samples", n_samples, 0)
        X = self._check_X(X)
        if X is None and n_samples is None:
            raise ValueError("Pass X, or n_samples to draw that many values from the root's distribution.")
        return self._sample(X, n_samples, random_state)

    def _check_estimator(self, estimator: "BaseLocalEstimator | None", family: type) -> None:
        """Check that ``estimator`` is ``None`` or a local estimator of ``family``, and take the
        ``supports_weighted_data`` tag from it, as the estimator decides whether the rows can be weighted."""
        if estimator is None:
            return
        if not isinstance(estimator, family):
            raise TypeError(f"estimator must be a {family.__name__}, but is a {type(estimator).__name__}.")
        self.set_tags(supports_weighted_data=estimator.get_tag("supports_weighted_data"))

    def _check_X(self, X: pd.DataFrame | None, method: str | None = None) -> pd.DataFrame | None:
        """Return ``X``'s columns in ``evidence_`` order, checking that they are the parents seen in fit. ``X=None``
        stands for a root's own distribution, unless ``method`` needs ``X``."""
        self.check_is_fitted()
        if X is None:
            if self.evidence_:
                raise ValueError(
                    f"{self.variable_!r} has parents {self.evidence_}, so pass their values in X; X=None is only for a "
                    "root. Its marginal distribution depends on its parents' distributions, so get it from the network."
                )
            if method is not None:
                raise ValueError(f"{method} needs X; for a root variable, pass a DataFrame without columns.")
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
        if self.get_tag("variable_type") == ["continuous"]:
            return pd.DataFrame(distribution.mean().to_numpy(), index=X.index, columns=[self.variable_])
        categories = pd.Series(distribution.categories)
        values = categories.to_numpy()[np.argmax(distribution.probs, axis=1)]
        dtype = categories.infer_objects().dtype
        return pd.DataFrame(values.reshape(-1, 1), index=X.index, columns=[self.variable_]).astype(dtype)

    def _log_likelihood(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Predict the distributions on a RangeIndex and put X's index back at the end, as skpro matches values
        # to rows by label, which mixes up rows with repeated labels.
        distribution = self._predict_proba(X.set_axis(pd.RangeIndex(len(X))))

        # Step 2: Score y with the log-probability under a discrete measure, and the log-density under any other. y goes
        # in as a numpy array, which skpro matches to rows by position.
        values = y.to_numpy()
        discrete = distribution.get_tag("distr:measuretype", "continuous", raise_error=False) == "discrete"
        scores = distribution.log_pmf(values) if discrete else distribution.log_pdf(values)
        return pd.DataFrame(scores.to_numpy(), index=X.index, columns=[self.variable_])

    def _sample(
        self, X: pd.DataFrame | None, n_samples: int | None, random_state: int | np.random.Generator | None
    ) -> pd.DataFrame:
        # Step 1: Repeat X's rows once per draw, or for a root without X, take n_samples rows without columns, and draw
        # one value for each, on a RangeIndex: skpro matches values to rows by label, which mixes up rows with repeated
        # labels.
        if X is None:
            rows = pd.DataFrame(index=pd.RangeIndex(n_samples))
        else:
            rows = X.iloc[np.tile(np.arange(len(X)), 1 if n_samples is None else n_samples)]
        samples = self._draw(rows.set_axis(pd.RangeIndex(len(rows))), random_state)

        # Step 2: Label the samples with X's index, and a level numbering the draws, and with the target's name.
        if X is not None:
            samples = samples.set_axis(X.index if n_samples is None else _sample_index(X.index, n_samples))
        return samples.set_axis([self.variable_], axis=1)

    def _draw(self, rows: pd.DataFrame, random_state: int | np.random.Generator | None) -> pd.DataFrame:
        """Draw one value of the target for each row of parent values, which come on a RangeIndex: with the predicted
        distribution's own sample() if it takes a random_state, as NominalDistribution's does, and otherwise, as for
        skpro's distributions, with the ppf of uniform values from the seeded generator."""
        distribution = self._predict_proba(rows)
        if "random_state" in inspect.signature(distribution.sample).parameters:
            return distribution.sample(random_state=random_state)
        uniform = np.random.default_rng(random_state).random((len(rows), 1))
        return pd.DataFrame(distribution.ppf(uniform).to_numpy(), dtype=float)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or self.is_fitted != other.is_fitted:
            return False
        if self.is_fitted:
            return self is other or self._fitted_equal(other)
        return _equal(self.get_params(deep=False), other.get_params(deep=False))

    def __hash__(self) -> int:
        # Constant per class, as fit changes the object but must not change its hash.
        return hash(type(self))

    def _fitted_equal(self, other: "BaseParameterization") -> bool:
        """Return whether this fitted object learned the same as ``other``, a fitted object of the same class. Classes
        that compare what they learned override it."""
        return False


class BaseLocalEstimator(BaseObject):
    """Base class for local estimators: estimators of one parameterization's parameters from one node's data.

    A local estimator is passed to its parameterization, e.g. ``TabularCPD(estimator=TabularBayesian())``, whose
    ``fit`` prepares the data and hands it to the estimator. Each parameterization has its own base class of local
    estimators, which fixes what the estimator gets and returns: ``BaseTabularEstimator`` gets the weighted counts and
    returns the table, ``BaseLinearGaussianEstimator`` gets the rows and returns the coefficients and the std, and
    ``BasePyroEstimator`` gets the node's Pyro model and returns posterior draws.
    Subclasses set a ``name`` and the other tags registered for local estimators in :mod:`pgmpy.registry`.
    """

    _tags = {
        "object_type": "local_estimator",
        "name": None,
        "parameterization": [],
        "supports_weighted_data": False,
        "python_dependencies": None,
    }

    def __init__(self) -> None:
        _check_estimator_deps(self)
        super().__init__()


def _equal(value: Any, other: Any) -> bool:
    """Return whether two parameter values are equal: numbers and numeric arrays within numpy.allclose's tolerance,
    other values of the same type, containers item by item, and skbase objects and sklearn estimators by their
    parameters. skbase's ``deep_equals`` would compare sklearn estimators by identity, and mishandles object and string
    arrays."""
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
        return value.keys() == other.keys() and all(_equal(value[key], other[key]) for key in value)
    if isinstance(value, (list, tuple)):
        return len(value) == len(other) and all(_equal(a, b) for a, b in zip(value, other))
    if isinstance(value, (BaseObject, SklearnEstimator)):
        return _equal(value.get_params(deep=False), other.get_params(deep=False))
    return value == other


def _check_integer(name: str, value: Any, minimum: int) -> None:
    """Check that ``value`` is an integer of at least ``minimum``, and not a bool."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, but is {value!r}.")


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


def _checked_state_names(state_names: dict | None) -> dict:
    """Return the given state names as lists, rejecting anything but a dict and repeated states."""
    if state_names is None:
        return {}
    if not isinstance(state_names, dict):
        raise TypeError(f"state_names must be a dict of {{variable: [states]}}, but is a {type(state_names).__name__}.")
    state_names = {variable: list(states) for variable, states in state_names.items()}
    for variable, states in state_names.items():
        if len(set(states)) != len(states):
            raise ValueError(f"Repeated state names for variable {variable!r}: {states}.")
    return state_names


def _states(columns: list[tuple[Hashable, pd.Series]], given: dict) -> dict:
    """Return the states of each variable in ``columns``, pairs of a name and its values: the variable's states in
    ``given``, checked state names that must include every value, else its values, sorted."""
    states = {}
    for name, values in columns:
        if name not in given:
            states[name] = sorted(values.unique())
        elif not set(values.unique()) <= set(given[name]):
            raise ValueError(f"Data contains unexpected states for variable: {name!r}.")
        else:
            states[name] = given[name]
    return states


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
