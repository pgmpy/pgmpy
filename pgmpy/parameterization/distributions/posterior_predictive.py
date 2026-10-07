import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.parameterization.distributions.nominal import _sample_index

BaseDistribution = _safe_import("skpro.distributions.base.BaseDistribution")
torch = _safe_import("torch")


class PosteriorPredictive(BaseDistribution):
    """Posterior predictive distribution of a Bayesian parameterization, such as ``PyroAdapter``: for each row, the
    mixture of the target's distribution over the posterior draws, with equal weights.

    It holds the fitted parameterization and the parents' values, and evaluates the target's distribution when a method
    needs it, so it can be subset like any skpro distribution. The density, or the probability for a discrete support,
    is the average over the draws, and 0 outside the support. The mean is the average of the draws' means, and the
    variance the average of their variances plus the variance of their means. The cdf is the average of the draws'
    cdfs, for families that have one in torch, with values clamped to the support's bounds; ``ppf`` is skpro's
    approximation from the cdf.

    Requires the optional dependencies ``pyro-ppl`` and ``skpro``.

    Parameters
    ----------
    parameterization : PyroAdapter
        A fitted parameterization with posterior draws, ``posterior_samples_``, which gives the target's torch
        distribution given the parents' values under some of the draws.
    X : pandas.DataFrame
        The parents' values, one row per row of the distribution, used by position; one row for a scalar distribution.
    index : pandas.Index or list, optional
        Row labels, a ``RangeIndex`` by default. Without ``index`` and ``columns``, the distribution is scalar.
    columns : pandas.Index or list, optional
        The column label, the target's name by default.

    Examples
    --------
    A ``fn`` with nothing to learn has a single draw, so its posterior predictive is ``fn``'s distribution:

    >>> import pandas as pd
    >>> import pyro.distributions as dist
    >>> from pgmpy.parameterization import PyroAdapter, PyroSVI
    >>> X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    >>> y = pd.Series([1.0, 3.0, 5.0], name="y")
    >>> line = lambda parents: dist.Normal(1.0 + 2.0 * parents["a"], 0.5)
    >>> node = PyroAdapter(line, estimator=PyroSVI(num_steps=1)).fit(X, y)
    >>> predictive = node.predict_proba(pd.DataFrame({"a": [0.0, 1.0]}))
    >>> predictive.mean()["y"].tolist()
    [1.0, 3.0]
    >>> predictive.var()["y"].tolist()
    [0.25, 0.25]
    """

    _tags = {
        "python_dependencies": "pyro-ppl",
        "distr:measuretype": "continuous",
        "capabilities:approx": ["ppf", "energy", "pdfnorm"],
        "capabilities:exact": ["pdf", "log_pdf", "pmf", "log_pmf", "mean", "var", "cdf"],
        "broadcast_init": "off",
    }

    def __init__(
        self,
        parameterization: Any,
        X: pd.DataFrame,
        index: pd.Index | list | None = None,
        columns: pd.Index | list | None = None,
    ) -> None:
        _check_soft_dependencies("skpro", obj=self)

        self.parameterization = parameterization
        self.X = X

        # A scalar distribution has one row of parents. An array distribution has one row per row of X, and the target
        # as its column.
        if index is None and columns is None:
            if len(X) != 1:
                raise ValueError(f"A scalar PosteriorPredictive needs X with one row, but X has {len(X)}.")
        else:
            index = pd.RangeIndex(len(X)) if index is None else index
            columns = [parameterization.variable_] if columns is None else columns
            if len(index) != len(X):
                raise ValueError(f"index needs one label per row of X, {len(X)}, but has {len(index)}.")
            if len(columns) != 1:
                raise ValueError("columns must contain exactly one column name.")

        super().__init__(index=index, columns=columns)
        if parameterization._discrete:
            self.set_tags(**{"distr:measuretype": "discrete"})

    def _torch_distribution(self) -> Any:
        """Return the target's torch distribution under every posterior draw, with batch shape (draws, rows)."""
        distribution = self.parameterization._distribution(self.X, self.parameterization.posterior_samples_)
        # Values outside the support are handled here, so torch mustn't reject them. Pyro's validation switch doesn't
        # reach a distribution that a plate expanded, which keeps the flag it was created with; its parameters were
        # checked then.
        distribution._validate_args = False
        return distribution

    def _values(self, x: ArrayLike) -> Any:
        """Return the queried values, one per row, as a 1-D tensor in torch's default dtype."""
        # A copy, as skpro passes read-only views, which torch warns about.
        return torch.tensor(np.asarray(x, dtype=float).reshape(-1), dtype=torch.get_default_dtype())

    def _log_density(self, x: ArrayLike) -> np.ndarray:
        """Return the log of the average density, or probability, of ``x`` over the draws: -inf outside the support."""
        if len(self.X) == 0:
            return np.empty(self.shape)
        distribution = self._torch_distribution()
        values = self._values(x)
        log_probs = torch.where(distribution.support.check(values), distribution.log_prob(values), -torch.inf)
        return _to_numpy(torch.logsumexp(log_probs, dim=0) - math.log(log_probs.shape[0]), self.shape)

    def _log_pdf(self, x: ArrayLike) -> np.ndarray:
        return self._log_density(x)

    def _log_pmf(self, x: ArrayLike) -> np.ndarray:
        return self._log_density(x)

    def _pdf(self, x: ArrayLike) -> np.ndarray:
        return np.exp(self._log_density(x))

    def _pmf(self, x: ArrayLike) -> np.ndarray:
        return np.exp(self._log_density(x))

    def _mean(self) -> np.ndarray:
        if len(self.X) == 0:
            return np.empty(self.shape)
        return _to_numpy(self._torch_distribution().mean.mean(dim=0), self.shape)

    def _var(self) -> np.ndarray:
        if len(self.X) == 0:
            return np.empty(self.shape)
        distribution = self._torch_distribution()
        variance = distribution.variance.mean(dim=0) + distribution.mean.var(dim=0, correction=0)
        return _to_numpy(variance, self.shape)

    def _cdf(self, x: ArrayLike) -> np.ndarray:
        if len(self.X) == 0:
            return np.empty(self.shape)
        distribution = self._torch_distribution()
        if type(distribution).cdf is torch.distributions.Distribution.cdf:
            raise NotImplementedError(
                f"{type(distribution).__name__} has no cdf in torch, so its PosteriorPredictive has none either."
            )
        # Clamped to the support's bounds, values below the support get 0, and values above it get 1.
        values = self._values(x)
        lower, upper = (
            getattr(distribution.support, "lower_bound", None),
            getattr(distribution.support, "upper_bound", None),
        )
        if lower is not None or upper is not None:
            values = values.clamp(min=lower, max=upper)
        return _to_numpy(distribution.cdf(values).mean(dim=0), self.shape)

    def sample(self, n_samples: int | None = None, random_state: int | np.random.Generator | None = None) -> Any:
        """Draw values from the distribution: each from the target's distribution under one posterior draw, picked at
        random.

        Parameters
        ----------
        n_samples : int, optional
            Number of draws, one if omitted.
        random_state : int, numpy.random.Generator or None, optional
            Seed or generator for this call; ``None`` draws fresh values. The global torch random state doesn't change.

        Returns
        -------
        float or pandas.DataFrame
            For a scalar distribution, one value, or with ``n_samples`` a DataFrame with one row per draw. For an array
            distribution, a DataFrame with its index and columns, preceded by a level numbering the draws when
            ``n_samples`` is given.
        """
        # Step 1: Pick one posterior draw for each value, with the seeded generator.
        rng = np.random.default_rng(random_state)
        rows = np.tile(np.arange(len(self.X)), 1 if n_samples is None else n_samples)
        draws = self.parameterization.posterior_samples_
        picked = rng.integers(len(next(iter(draws.values()))) if draws else 1, size=len(rows))

        # Step 2: Evaluate the target's distribution once for each draw picked, on the rows that picked it, and sample
        # it, in a copy of the torch random state seeded from the same generator.
        values = np.empty(len(rows))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(rng.integers(2**63)))
            for draw in np.unique(picked):
                positions = np.flatnonzero(picked == draw)
                one_draw = {name: samples[draw : draw + 1] for name, samples in draws.items()}
                distribution = self.parameterization._distribution(self.X.iloc[rows[positions]], one_draw)
                values[positions] = distribution.sample()[0].detach().cpu().numpy()

        # Step 3: Return one value for a single draw from a scalar distribution, and a DataFrame otherwise.
        if self.ndim == 0:
            return float(values[0]) if n_samples is None else pd.DataFrame(values.reshape(-1, 1))
        index = self.index if n_samples is None else _sample_index(self.index, n_samples)
        return pd.DataFrame(values.reshape(-1, 1), index=index, columns=self.columns)

    def _subset_params(self, rowidx: Any, colidx: Any, coerce_scalar: bool = False) -> dict[str, Any]:
        # There is only one column, so nothing to subset, but an out-of-range position must raise IndexError.
        if colidx is not None:
            self.columns[colidx]

        X = self.X
        if rowidx is not None:
            X = X.iloc[[rowidx]] if isinstance(rowidx, (int, np.integer)) else X.iloc[rowidx]
        return {"parameterization": self.parameterization, "X": X}


def _to_numpy(values: Any, shape: tuple) -> np.ndarray:
    """Return a tensor of one value per row as a float64 array of the distribution's shape."""
    return values.detach().cpu().numpy().astype(float).reshape(shape)
