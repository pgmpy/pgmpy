import sys
from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _safe_import
from sklearn.linear_model import LinearRegression

from pgmpy.parameterization._base import BaseParameter, _checked_evidence, _parent_order

Normal = _safe_import("skpro.distributions.Normal")


class LinearGaussianCPD(BaseParameter):
    """Linear Gaussian conditional probability distribution of a continuous variable given continuous parents.

    ``fit(X, y)`` estimates ``y | X ~ N(beta_[0] + X @ beta_[1:], std_**2)`` by least squares, with the same estimates
    as ``LinearGaussianMLE``. Sample weights count as frequencies, so integer weights give the fit of each row repeated
    that many times. Parents are sorted by name. Fit a root variable with ``X=None``; its ``beta_`` holds only its
    mean. ``from_values`` creates a fitted instance from known coefficients instead.

    Without ``X``, ``predict_proba`` gives the marginal distribution of the target that the model implies for the parent
    values seen in ``fit``: ``N(beta_[0] + b @ mu, b @ Sigma @ b + std_**2)`` with ``b = beta_[1:]``, where ``mu`` and
    ``Sigma`` are the mean and covariance of those parent values. It is exact when the parents are jointly Gaussian. For
    an instance from ``from_values``, only a root variable has one. Equality compares the conditional distributions, not
    these marginals.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    std_estimator : {"unbiased", "mle"}, default="unbiased"
        Estimate of ``std_``, as in ``LinearGaussianMLE``. ``"unbiased"`` divides the sum of squared residuals by the
        number of rows minus the number of independent coefficients: the intercept plus the rank of the parent data,
        which differs from ``LinearGaussianMLE``'s count of the parents only for collinear parents. ``"mle"`` divides by
        the number of rows. With sample weights, their total replaces the number of rows. The covariance ``Sigma`` of
        the parents likewise divides by the number of rows minus one, or by the number of rows.

    Attributes
    ----------
    beta_ : numpy.ndarray of shape (1 + n_parents,)
        Intercept, then one coefficient per parent in ``evidence_`` order. A root variable has only its mean.
    std_ : float
        Standard deviation of the target given its parents.
    variable_ : hashable
        Name of the target variable.
    evidence_ : list
        Names of the parent variables, sorted.

    Warnings
    --------
    Experimental: the API of ``pgmpy.parameterization`` may change in any release without a deprecation period.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.parameterization.cpds import LinearGaussianCPD
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame({"A": rng.normal(size=1000), "B": rng.normal(size=1000)})
    >>> y = pd.Series(1 + 2 * X["A"] - 3 * X["B"] + rng.normal(scale=0.5, size=1000), name="y")
    >>> cpd = LinearGaussianCPD().fit(X, y)
    >>> cpd.beta_.round(1).tolist()
    [1.0, 2.0, -3.0]
    >>> round(cpd.std_, 1)
    0.5
    >>> cpd.sample(pd.DataFrame({"A": [0.0], "B": [0.0]}), random_state=0).round(2)["y"].tolist()
    [1.19]

    Without ``X``, ``predict_proba`` gives the marginal distribution implied for the parent values seen in ``fit``:

    >>> marginal = cpd.predict_proba()
    >>> round(float(marginal.mean()), 1), round(float(marginal.var()), 1)
    (1.2, 13.7)

    A root variable is fitted without parents, or created from known values with the arguments of
    ``pgmpy.factors.continuous.LinearGaussianCPD``:

    >>> root = LinearGaussianCPD.from_values("A", [0.0], 1.0)
    >>> float(root.predict_proba().mean())
    0.0
    """

    _tags = {
        "variable_type": "continuous",
        "parent_type": "continuous",
        "supports_weighted_data": True,
        "python_dependencies": "skpro",
    }

    def __init__(self, std_estimator: str = "unbiased") -> None:
        self.std_estimator = std_estimator
        super().__init__()

    @classmethod
    def from_values(
        cls, variable: Hashable, beta: ArrayLike, std: float, evidence: list | tuple | None = None
    ) -> "LinearGaussianCPD":
        """Create a fitted LinearGaussianCPD from known coefficients.

        The arguments are those of ``pgmpy.factors.continuous.LinearGaussianCPD``, in the same order.

        Parameters
        ----------
        variable : hashable
            Name of the target variable.
        beta : array-like of shape (1 + len(evidence),)
            Intercept, then one coefficient per parent in ``evidence`` order. All real and finite.
        std : float
            Standard deviation of ``variable`` given its parents: positive and finite.
        evidence : list or tuple, optional
            Names of the parent variables. ``None`` for a root variable.

        Returns
        -------
        LinearGaussianCPD
            A fitted instance. Its parents are sorted by name, with ``beta`` reordered to match.
        """
        evidence = _checked_evidence(variable, evidence)
        if np.iscomplexobj(beta):
            raise ValueError(f"beta must hold real numbers, but is {beta!r}.")
        beta = np.array(beta, dtype=float)
        if beta.shape != (1 + len(evidence),) or not np.isfinite(beta).all():
            raise ValueError(
                f"beta must hold a finite intercept and a finite coefficient per parent in {evidence}, shape "
                f"({1 + len(evidence)},), but is {beta.tolist()}."
            )
        if not isinstance(std, Real) or not 0 < std <= sys.float_info.max:
            raise ValueError(f"std must be a positive, finite number, but is {std!r}.")

        order = _parent_order(evidence)
        cpd = cls()
        cpd.variable_ = variable
        cpd.evidence_ = [evidence[position] for position in order]
        cpd.variable_type_ = "continuous"
        cpd.beta_ = beta[[0, *(1 + position for position in order)]]
        cpd.std_ = float(std)
        cpd._marginal = None if evidence else (cpd.beta_[0], cpd.std_)
        cpd._is_fitted = True
        return cpd

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        if self.std_estimator not in ("unbiased", "mle"):
            raise ValueError(f"std_estimator must be 'unbiased' or 'mle', but is {self.std_estimator!r}.")
        # Rows without weight don't count, and dropping them keeps their values out of the sums of squares below,
        # where they could overflow.
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        positive = weights > 0
        parents, target = X.to_numpy(dtype=float)[positive], y.iloc[:, 0].to_numpy(dtype=float)[positive]
        weights = weights[positive]
        if np.ptp(target) == 0:
            raise ValueError(
                f"{self.variable_!r} is constant, so its std is 0, but a LinearGaussianCPD needs a positive std."
            )

        if self.evidence_:
            # An array, not the DataFrame, so that sklearn accepts any hashable column names.
            regression = LinearRegression().fit(
                parents, target, sample_weight=None if sample_weight is None else weights
            )
            self.beta_ = np.concatenate([[regression.intercept_], regression.coef_])
            # rank_ is the rank of the centred parent data, which leaves out the intercept.
            n_coefficients = 1 + regression.rank_
        else:
            self.beta_ = np.array([np.average(target, weights=weights)])
            n_coefficients = 1

        total = weights.sum()
        # Weights meant to total n_coefficients can sum to just above it, so allow for rounding.
        if self.std_estimator == "unbiased" and total <= n_coefficients * (1 + 1e-9):
            raise ValueError(
                f"The unbiased std needs a total weight, or number of rows, above the {n_coefficients} independent "
                f"coefficients, but it is {total:g}. Fit more data, or use std_estimator='mle'."
            )
        means = self.beta_[0] + parents @ self.beta_[1:]
        ddof, marginal_ddof = (0, 0) if self.std_estimator == "mle" else (n_coefficients, 1)
        self.std_ = float(np.sqrt(np.sum(weights * (target - means) ** 2) / (total - ddof)))
        if not 0 < self.std_ < np.inf:
            raise ValueError(
                f"The fitted std of {self.variable_!r} is {self.std_:g}, but must be positive and finite. A target "
                "that is constant given its parents has a std of 0."
            )

        # b @ mu and b @ Sigma @ b are the mean and variance of the fitted means, b @ X, over the rows seen in fit.
        mean = np.average(means, weights=weights)
        variance = np.sum(weights * (means - mean) ** 2) / (total - marginal_ddof) + self.std_**2
        self._marginal = (mean, float(np.sqrt(variance)))

    def _mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.beta_[0] + X.to_numpy(dtype=float) @ self.beta_[1:]

    def _predict_proba(self, X: pd.DataFrame | None) -> "Normal":
        if X is not None:
            return Normal(mu=self._mean(X).reshape(-1, 1), sigma=self.std_, index=X.index, columns=[self.variable_])
        if self._marginal is None:
            raise ValueError(
                "A LinearGaussianCPD with parents created by from_values has no data on its parents, so its marginal "
                "distribution is unknown; pass X."
            )
        return Normal(mu=self._marginal[0], sigma=self._marginal[1])
