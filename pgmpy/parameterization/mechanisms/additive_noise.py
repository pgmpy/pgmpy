from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.utils.dependencies import _safe_import
from sklearn.model_selection import BaseCrossValidator, check_cv

from pgmpy.parameterization._base import BaseParameterization
from pgmpy.parameterization.adapters import DeterministicAdapter, DistributionAdapter
from pgmpy.parameterization.cpds import LinearGaussianCPD
from pgmpy.parameterization.mechanisms._base import BaseMechanism

MeanScale = _safe_import("skpro.distributions.MeanScale")

if TYPE_CHECKING:
    from skpro.distributions.base import BaseDistribution


class AdditiveNoiseMechanism(BaseMechanism):
    """Additive noise mechanism: Y = f(Pa) + U, with noise U independent of the parents.

    f is the point prediction of ``function``, any parameterization of a continuous target, e.g. ``SklearnAdapter``
    wrapping a regressor; any noise of its own, such as ``LinearGaussianCPD``'s ``std_``, is ignored. The noise is
    ``noise``, any parameterization of a continuous target that gives a continuous distribution, fitted as a root on the
    residuals y - f(x): by default ``LinearGaussianCPD()``, i.e. Gaussian noise with a fitted mean and standard
    deviation, or e.g. a ``SkproAdapter`` root for another family. The residuals are in-sample, or with ``cv`` out of
    fold, which a flexible function needs, as its in-sample residuals understate the noise.

    ``predict_proba`` gives each row the noise's distribution shifted by f(x), ``predict`` gives f(x) + E[U], and
    ``log_likelihood`` and ``sample`` follow from them. ``abduct`` returns y - f(x), and ``predict(X, noise=u)`` returns
    f(x) + u. ``from_values`` creates a fitted mechanism from a known function and a known noise distribution, without
    data. The tags follow the slots: the parent types come from ``function``, weights need both slots to take them, and
    a linear function with Gaussian noise supports exact inference.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    function : BaseParameterization
        Parameterization of a continuous target, whose ``predict`` gives f.
    noise : BaseParameterization, optional
        Parameterization of a continuous target that gives a continuous distribution, fitted as a root on the
        residuals. ``None`` for ``LinearGaussianCPD()``.
    cv : int, cross-validation generator or iterable, optional
        How to split the rows for out-of-fold residuals, as in ``sklearn.model_selection.cross_val_predict``. ``None``
        for in-sample residuals.

    Attributes
    ----------
    function_ : BaseParameterization
        The fitted clone of ``function``.
    noise_ : BaseParameterization
        The clone of ``noise`` fitted as a root on the residuals.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from pgmpy.parameterization import AdditiveNoiseMechanism, SklearnAdapter
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame({"a": rng.normal(size=500)})
    >>> y = pd.Series(1 + 2 * X["a"] + rng.normal(scale=0.5, size=500), name="y")
    >>> mechanism = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression())).fit(X, y)
    >>> unit = pd.DataFrame({"a": [0.0]})
    >>> mechanism.predict(unit).round(1)["y"].tolist()
    [1.0]

    A counterfactual: the noise behind an observed value, and the value it gives had the parent been 1:

    >>> noise = mechanism.abduct(unit, [1.5])
    >>> mechanism.predict(unit.assign(a=1.0), noise=noise).round(1)["y"].tolist()
    [3.5]
    """

    _tags = {
        "name": "additive_noise_mechanism",
        "variable_type": ["continuous"],
        "assumption:additive_noise": True,
        "python_dependencies": "skpro",
    }

    def __init__(
        self,
        function: BaseParameterization,
        noise: BaseParameterization | None = None,
        cv: int | BaseCrossValidator | Iterable | None = None,
    ) -> None:
        self.function = function
        self.noise = noise
        self.cv = cv
        super().__init__()

        noise = LinearGaussianCPD() if noise is None else noise
        for name, slot in (("function", function), ("noise", noise)):
            if not isinstance(slot, BaseParameterization) or slot.get_tag("variable_type") != ["continuous"]:
                raise TypeError(f"{name} must be a parameterization of a continuous target, but is {slot!r}.")
        if not noise.get_tag("capability:distribution"):
            raise TypeError(f"noise must give a distribution, but {type(noise).__name__} gives point predictions.")
        # The parents are f's, and the noise assumptions follow from both slots.
        linear, gaussian = function.get_tag("assumption:linearity"), noise.get_tag("assumption:gaussian_noise")
        self.set_tags(
            **{
                "parent_data_types": function.get_tag("parent_data_types"),
                "supports_weighted_data": function.get_tag("supports_weighted_data")
                and noise.get_tag("supports_weighted_data"),
                "assumption:linearity": linear,
                "assumption:gaussian_noise": gaussian,
                "capability:exact_inference": linear and gaussian,
            }
        )

    @classmethod
    def from_values(
        cls,
        variable: Hashable,
        function: Callable[[pd.DataFrame], ArrayLike],
        noise: "BaseDistribution",
        evidence: list | tuple | None = None,
    ) -> "AdditiveNoiseMechanism":
        """Create a fitted AdditiveNoiseMechanism from a known function and a known noise distribution, without data.

        Parameters
        ----------
        variable : hashable
            Name of the target variable.
        function : callable
            f, given the parents, as for ``DeterministicAdapter``.
        noise : skpro distribution
            The noise's distribution, scalar and continuous, e.g. ``skpro.distributions.Laplace(mu=0.0, scale=1.0)``.
        evidence : list or tuple, optional
            Names of the parent variables. ``None`` for a root variable.

        Returns
        -------
        AdditiveNoiseMechanism
            A fitted instance. Its parents are sorted by name.
        """
        mechanism = cls(DeterministicAdapter(function), DistributionAdapter(noise))
        mechanism.function_ = DeterministicAdapter.from_values(variable, function, evidence)
        mechanism.noise_ = DistributionAdapter.from_values(variable, noise)
        _check_continuous(mechanism.noise_)
        mechanism.variable_ = variable
        mechanism.evidence_ = mechanism.function_.evidence_
        mechanism._is_fitted = True
        return mechanism

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Fit f, and compute the residuals: in-sample, or out of fold with cv, so that a flexible f doesn't
        # leave residuals it has memorized.
        self.function_ = self.function.clone().fit(X, y, sample_weight)
        if self.cv is None:
            fitted = self.function_.predict(X).iloc[:, 0].to_numpy(dtype=float)
        else:
            fitted = np.empty(len(y))
            for train, test in check_cv(self.cv).split(X):
                weights = None if sample_weight is None else sample_weight[train]
                fold = self.function.clone().fit(X.iloc[train], y.iloc[train], weights)
                fitted[test] = fold.predict(X.iloc[test]).iloc[:, 0].to_numpy(dtype=float)
        residuals = pd.Series(y.iloc[:, 0].to_numpy(dtype=float) - fitted, index=y.index, name=self.variable_)
        if np.ptp(residuals) == 0:
            raise ValueError(
                "The residuals are all equal: the function reproduces its training data, as KNN with one neighbour "
                "does. Pass cv to fit the noise on out-of-fold residuals."
            )

        # Step 2: Fit the noise as a root on the residuals, Gaussian by default, and check that it's continuous.
        noise = LinearGaussianCPD() if self.noise is None else self.noise.clone()
        self.noise_ = noise.fit(None, residuals, sample_weight)
        _check_continuous(self.noise_)

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        # Step 1: Without X, which only a root gets, return the distribution of one row, as a scalar.
        if X is None:
            return self._predict_proba(pd.DataFrame(index=[0])).iat[0, 0]

        # Step 2: Shift the noise's distribution, repeated to the rows of X, by f(X).
        noise = self.noise_.predict_proba(pd.DataFrame(index=X.index))
        means = self.function_.predict(X).to_numpy(dtype=float)
        return MeanScale(d=noise, mu=means, sigma=1.0, index=X.index, columns=[self.variable_])

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.function_.predict(X) + self.noise_.predict(pd.DataFrame(index=X.index)).to_numpy()

    def _forward(self, X: pd.DataFrame, noise: np.ndarray) -> pd.DataFrame:
        return self.function_.predict(X) + noise.reshape(-1, 1)

    def _abduct(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        residuals = y - self.function_.predict(X).iloc[:, 0].to_numpy(dtype=float)
        return pd.DataFrame(residuals.reshape(-1, 1), index=X.index, columns=[self.variable_])

    def _sample_noise(self, n_samples: int, random_state: int | np.random.Generator | None) -> np.ndarray:
        return self.noise_.sample(n_samples=n_samples, random_state=random_state).iloc[:, 0].to_numpy(dtype=float)


def _check_continuous(noise: BaseParameterization) -> None:
    """Raise unless the fitted noise's distribution is continuous, as the noise on a continuous target must be."""
    distribution = noise.predict_proba()
    if distribution.get_tag("distr:measuretype", "continuous", raise_error=False) == "discrete":
        raise ValueError(
            f"The noise's distribution, a {type(distribution).__name__}, is discrete, but the noise added to a "
            "continuous target must be continuous."
        )
