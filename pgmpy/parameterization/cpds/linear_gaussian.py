import sys
from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _safe_import

from pgmpy.parameterization._base import BaseParameterization, _checked_evidence, _parent_order
from pgmpy.parameterization.cpds.linear_gaussian_estimators import BaseLinearGaussianEstimator, LinearGaussianOLS

Normal = _safe_import("skpro.distributions.Normal")


class LinearGaussianCPD(BaseParameterization):
    """Linear Gaussian conditional probability distribution of a continuous variable given continuous parents.

    ``fit(X, y)`` estimates ``y | X ~ N(beta_[0] + X @ beta_[1:], std_**2)`` with ``estimator``, by least squares by
    default. Sample weights count as frequencies, so integer weights give the fit of each row repeated that many times.
    Parents are sorted by name. Fit a root variable with ``X=None``; its ``beta_`` holds only its mean.
    ``from_values`` creates a fitted instance from known coefficients instead.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    estimator : BaseLinearGaussianEstimator, optional
        Estimates the coefficients and the standard deviation from the data, e.g.
        ``LinearGaussianOLS(std_estimator="mle")``. ``None`` for ``LinearGaussianOLS()``.

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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.parameterization import LinearGaussianCPD
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
    >>> cpd.predict(pd.DataFrame({"A": [0.0], "B": [1.0]})).round(1)["y"].tolist()
    [-2.0]
    >>> round(float(cpd.log_likelihood(X, y)["y"].mean()), 2)
    -0.74

    A root variable is fitted without parents, or created from known values:

    >>> root = LinearGaussianCPD.from_values("A", [0.0], 1.0)
    >>> float(root.predict_proba().mean())
    0.0
    """

    _tags = {
        "name": "linear_gaussian_cpd",
        "variable_type": ["continuous"],
        "parent_data_types": ["continuous"],
        "supports_weighted_data": True,
        "capability:exact_inference": True,
        "assumption:linearity": True,
        "assumption:additive_noise": True,
        "assumption:gaussian_noise": True,
        "python_dependencies": "skpro",
    }

    def __init__(self, estimator: BaseLinearGaussianEstimator | None = None) -> None:
        self.estimator = estimator
        super().__init__()

        # The estimator decides whether the rows can be weighted.
        if estimator is not None:
            if not isinstance(estimator, BaseLinearGaussianEstimator):
                raise TypeError(
                    "estimator must be a linear Gaussian estimator, such as LinearGaussianOLS(), but is a "
                    f"{type(estimator).__name__}."
                )
            self.set_tags(supports_weighted_data=estimator.get_tag("supports_weighted_data"))

    @classmethod
    def from_values(
        cls, variable: Hashable, beta: ArrayLike, std: float, evidence: list | tuple | None = None
    ) -> "LinearGaussianCPD":
        """Create a fitted LinearGaussianCPD from known coefficients.

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
        # Step 1: Check the parents, the coefficients and the std.
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

        # Step 2: Sort the parents by name, reorder beta to match, and create the fitted instance.
        order = _parent_order(evidence)
        cpd = cls()
        cpd.variable_ = variable
        cpd.evidence_ = [evidence[position] for position in order]
        cpd.beta_ = beta[[0, *(1 + position for position in order)]]
        cpd.std_ = float(std)
        cpd._is_fitted = True
        return cpd

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Drop the rows with weight 0: they don't count, and dropping them keeps their values out of the sums of
        # squares, where they could overflow. A constant target has no positive std.
        parents, target = X.to_numpy(dtype=float), y.iloc[:, 0].to_numpy(dtype=float)
        if sample_weight is not None:
            positive = sample_weight > 0
            parents, target, sample_weight = parents[positive], target[positive], sample_weight[positive]
        if np.ptp(target) == 0:
            raise ValueError(
                f"{self.variable_!r} is constant, so its std is 0, but a LinearGaussianCPD needs a positive std."
            )

        # Step 2: Estimate the coefficients and the std, which must be positive and finite.
        estimator = LinearGaussianOLS() if self.estimator is None else self.estimator
        self.beta_, self.std_ = estimator.estimate(parents, target, sample_weight)
        if not 0 < self.std_ < np.inf:
            raise ValueError(
                f"The fitted std of {self.variable_!r} is {self.std_:g}, but must be positive and finite. A target "
                "that is constant given its parents has a std of 0."
            )

    def _predict_proba(self, X: pd.DataFrame | None) -> "Normal":
        # Step 1: Without X, return a root's distribution.
        if X is None:
            return Normal(mu=self.beta_[0], sigma=self.std_)

        # Step 2: Give each row the Normal around its fitted mean.
        means = self.beta_[0] + X.to_numpy(dtype=float) @ self.beta_[1:]
        return Normal(mu=means.reshape(-1, 1), sigma=self.std_, index=X.index, columns=[self.variable_])

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or not (self.is_fitted and other.is_fitted):
            return super().__eq__(other)
        return (
            (self.variable_, self.evidence_) == (other.variable_, other.evidence_)
            and np.allclose(self.beta_, other.beta_)
            and np.allclose(self.std_, other.std_)
        )

    __hash__ = BaseParameterization.__hash__
