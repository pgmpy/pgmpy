from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _safe_import

from pgmpy.parameterization._base import BaseParameterization, _checked_evidence, _parent_order
from pgmpy.utils._warnings import _warn_external

Delta = _safe_import("skpro.distributions.Delta")


class DeterministicAdapter(BaseParameterization):
    """Parameterization of a variable that is a known function of its parents: Y = f(Pa), with no noise.

    Each row of ``X`` gets the value the function gives it, as skpro's ``Delta``, so ``predict`` and ``sample`` return
    the values themselves. The function is given, not learned: ``from_values`` creates a fitted instance without data,
    e.g. for the known function of a simulated mechanism, and ``fit`` only records the target and its parents, with a
    warning that it ignores their values. ``log_likelihood`` is 0 where ``y`` matches the function's value within
    ``numpy.isclose``'s default tolerance, and minus infinity elsewhere; the ``Delta`` from ``predict_proba`` compares
    exactly.

    The function gets the parents as a DataFrame under their own names, sorted by name as in ``evidence_``, and a root's
    function gets rows without columns. ``pickle`` and ``joblib.dump`` can't save a lambda, so to save a model that
    holds one, use ``cloudpickle``, or a named function.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    function : callable
        Takes the parents as a DataFrame and returns one finite value per row, as an array, a Series or a one-column
        DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.parameterization import DeterministicAdapter
    >>> def bmi(X):
    ...     return X["weight"] / X["height"] ** 2
    >>> node = DeterministicAdapter.from_values("bmi", bmi, evidence=["height", "weight"])
    >>> node.predict(pd.DataFrame({"height": [1.8, 1.6], "weight": [81.0, 64.0]}))["bmi"].round(1).tolist()
    [25.0, 25.0]
    """

    _tags = {
        "name": "deterministic_adapter",
        "variable_type": ["continuous"],
        # The data is ignored, so none is needed, and weights change nothing.
        "supports_weighted_data": True,
        "requires_data": False,
        "python_dependencies": "skpro",
    }

    def __init__(self, function: Callable[[pd.DataFrame], ArrayLike]) -> None:
        self.function = function
        super().__init__()

        if not callable(function):
            raise TypeError(f"function must be callable, but is a {type(function).__name__}.")

    @classmethod
    def from_values(
        cls, variable: Hashable, function: Callable[[pd.DataFrame], ArrayLike], evidence: list | tuple | None = None
    ) -> "DeterministicAdapter":
        """Create a fitted DeterministicAdapter without data.

        Parameters
        ----------
        variable : hashable
            Name of the target variable.
        function : callable
            The target's value given its parents, as for ``DeterministicAdapter``.
        evidence : list or tuple, optional
            Names of the parent variables. ``None`` for a root variable.

        Returns
        -------
        DeterministicAdapter
            A fitted instance. Its parents are sorted by name.
        """
        evidence = _checked_evidence(variable, evidence)
        adapter = cls(function)
        adapter.variable_ = variable
        adapter.evidence_ = [evidence[position] for position in _parent_order(evidence)]
        adapter.variable_type_ = "continuous"
        adapter._is_fitted = True
        return adapter

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        _warn_external(
            "DeterministicAdapter.fit ignores the data: the function is given, not learned, so fit only records the "
            "target and its parents."
        )

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        # Step 1: Without X, which only a root gets, return the Delta of one row, as a scalar.
        if X is None:
            return self._predict_proba(pd.DataFrame(index=[0])).iat[0, 0]

        # Step 2: Evaluate the function on the rows, and check that it gives one finite value per row.
        values = np.asarray(self.function(X), dtype=float).reshape(-1)
        if values.shape != (len(X),):
            raise ValueError(f"function must return one value per row of X, {len(X)}, but returned {values.size}.")
        invalid = np.flatnonzero(~np.isfinite(values))
        if invalid.size:
            raise ValueError(
                f"function returned values that aren't finite for the rows at positions {invalid.tolist()}."
            )
        return Delta(c=values.reshape(-1, 1), index=X.index, columns=[self.variable_])

    def _log_likelihood(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        # A value counts as the function's within numpy.isclose's tolerance, as values computed in other ways, e.g. in
        # float32 or with the terms reordered, rarely match exactly.
        values = self._predict(X).iloc[:, 0].to_numpy()
        matches = np.isclose(y.iloc[:, 0].to_numpy(dtype=float), values)
        return pd.DataFrame(np.where(matches, 0.0, -np.inf).reshape(-1, 1), index=X.index, columns=[self.variable_])
