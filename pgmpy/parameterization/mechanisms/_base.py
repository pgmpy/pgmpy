import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pgmpy.parameterization._base import BaseParameterization, _check_n_samples
from pgmpy.parameterization.distributions.nominal import _sample_index


class BaseMechanism(BaseParameterization):
    """Base class for mechanisms: parameterizations Y = g(Pa, U), with noise U independent of the parents.

    A mechanism has the main API of every parameterization, so it can stand wherever a CPD can. On top of it,
    ``sample_noise`` draws noise, ``abduct`` recovers the noise behind observed values, and ``predict(X, noise=...)``
    runs the forward pass: the target's value given the parents and the noise. These are the steps of a counterfactual:
    abduct a unit's noise, change its parents, and predict with the same noise. ``sample`` draws noise and runs the
    forward pass, so ``sample(X, random_state=s)`` equals ``predict(X, noise=sample_noise(len(X), random_state=s))``.

    Subclasses implement ``_fit`` and ``_predict_proba``, as every parameterization does, and ``_sample_noise``,
    ``_forward`` and ``_abduct``.
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
        X = self._check_X(X)
        if X is None:
            raise ValueError("predict needs X; for a root variable, pass a DataFrame without columns.")
        return self._forward(X, _check_values(noise, X, "noise"))

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
        _check_n_samples(n_samples)
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
        X = self._check_X(X)
        if X is None:
            raise ValueError("abduct needs X; for a root variable, pass a DataFrame without columns.")
        return self._abduct(X, _check_values(y, X, "y"))

    def _sample(
        self, X: pd.DataFrame | None, n_samples: int | None, random_state: int | np.random.Generator | None
    ) -> pd.DataFrame:
        # Step 1: Repeat X's rows once per draw, or for a root without X, take n_samples rows without columns, and draw
        # one noise value for each.
        if X is None:
            rows = pd.DataFrame(index=pd.RangeIndex(n_samples))
        else:
            rows = X.iloc[np.tile(np.arange(len(X)), 1 if n_samples is None else n_samples)]
        noise = self._sample_noise(len(rows), random_state)

        # Step 2: Run the forward pass by position, and label the samples as every parameterization does: with X's
        # index, and a level numbering the draws when n_samples is given.
        samples = self._forward(rows.set_axis(pd.RangeIndex(len(rows))), noise)
        if X is not None:
            samples = samples.set_axis(X.index if n_samples is None else _sample_index(X.index, n_samples))
        return samples

    def _sample_noise(self, n_samples: int, random_state: int | np.random.Generator | None) -> np.ndarray:
        raise NotImplementedError

    def _forward(self, X: pd.DataFrame, noise: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError

    def _abduct(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError


def _check_values(values: ArrayLike | pd.Series | pd.DataFrame, X: pd.DataFrame, name: str) -> np.ndarray:
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
