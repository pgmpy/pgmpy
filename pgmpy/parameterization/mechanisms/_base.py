import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pgmpy.parameterization._base import BaseParameterization, _check_integer


class BaseMechanism(BaseParameterization):
    """Base class for mechanisms: parameterizations Y = g(Pa, U), with noise U independent of the parents.

    A mechanism has the main API of every parameterization, so it can stand wherever a CPD can. On top of it,
    ``sample_noise`` draws noise, ``abduct`` recovers the noise behind observed values, and ``predict(X, noise=...)``
    runs the forward pass: the target's value given the parents and the noise. These are the steps of a counterfactual:
    abduct a unit's noise, change its parents, and predict with the same noise. ``sample`` draws noise and runs the
    forward pass, so ``sample(X, random_state=s)`` equals ``predict(X, noise=sample_noise(len(X), random_state=s))``.

    Subclasses implement ``_fit`` and ``_predict_proba``, as every parameterization does, and ``_sample_noise``,
    ``_forward`` and ``_abduct``. ``_forward`` and ``_abduct`` get ``X`` in ``evidence_`` order, with ``X``'s own labels
    from ``predict`` and ``abduct`` and on a RangeIndex from ``sample``, and the noise or ``y`` as a 1-D float array by
    position. Both return one column named after the target, with ``X``'s index.
    """

    def predict(self, X: pd.DataFrame, noise: ArrayLike | pd.Series | pd.DataFrame | None = None) -> pd.DataFrame:
        """Return a point prediction of the target for each row of ``X``, or with ``noise``, the forward pass.

        Parameters
        ----------
        X : pandas.DataFrame
            Parent values, as for ``predict_proba``; for a root, a DataFrame without columns.
        noise : array-like, pandas.Series or pandas.DataFrame, optional
            One noise value per row of ``X``: an array by position, or a pandas object with ``X``'s index, such as the
            output of ``abduct``. Without it, the point prediction, as for every parameterization.

        Returns
        -------
        pandas.DataFrame
            One column named after the target, with ``X``'s index.
        """
        if noise is None:
            return super().predict(X)
        X = self._check_X(X, "predict")
        return self._forward(X, _checked_values(noise, X, "noise"))

    def sample_noise(self, n_samples: int, random_state: int | np.random.Generator | None = None) -> np.ndarray:
        """Draw noise values, which don't depend on the parents.

        Parameters
        ----------
        n_samples : int
            Number of values to draw.
        random_state : int, numpy.random.Generator or None, optional
            Seed or generator for this call; ``None`` draws fresh values.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The noise values, which ``predict(X, noise=...)`` takes by position.
        """
        self.check_is_fitted()
        _check_integer("n_samples", n_samples, 0)
        return self._sample_noise(n_samples, random_state)

    def abduct(self, X: pd.DataFrame, y: ArrayLike | pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Return the noise values that explain the observed ``y`` given the parents ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Parent values, as for ``predict``.
        y : array-like, pandas.Series or pandas.DataFrame
            Observed target values, one per row of ``X``: an array by position, or a pandas object with ``X``'s index.

        Returns
        -------
        pandas.DataFrame
            One column named after the target, with ``X``'s index.
        """
        X = self._check_X(X, "abduct")
        return self._abduct(X, _checked_values(y, X, "y"))

    def _draw(self, rows: pd.DataFrame, random_state: int | np.random.Generator | None) -> pd.DataFrame:
        return self._forward(rows, self._sample_noise(len(rows), random_state))

    def _sample_noise(self, n_samples: int, random_state: int | np.random.Generator | None) -> np.ndarray:
        raise NotImplementedError

    def _forward(self, X: pd.DataFrame, noise: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError

    def _abduct(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError


def _checked_values(values: ArrayLike | pd.Series | pd.DataFrame, X: pd.DataFrame, name: str) -> np.ndarray:
    """Return one finite number per row of ``X``, as a 1-D float array, from an array by position or a pandas object
    with ``X``'s index."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must have a single column, but has {values.shape[1]}.")
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series) and not values.index.equals(X.index):
        raise ValueError(f"{name} must have X's index, or be an array matched to X's rows by position.")
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (len(X),):
        raise ValueError(f"{name} needs one value per row of X, {len(X)}, but has {values.size}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must hold finite numbers.")
    return values
