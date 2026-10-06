from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from skbase.utils.dependencies import _safe_import
from sklearn.base import clone

from pgmpy.parameterization._base import BaseParameter, _plain
from pgmpy.parameterization.adapters.sklearn import _features
from pgmpy.utils import preprocess_data

DummyProbaRegressor = _safe_import("skpro.regression.dummy.DummyProbaRegressor")
Empirical = _safe_import("skpro.distributions.Empirical")
Normal = _safe_import("skpro.distributions.Normal")

if TYPE_CHECKING:
    from skpro.regression.base import BaseProbaRegressor


class SkproAdapter(BaseParameter):
    """Parameterization from a skpro probabilistic regressor.

    Each row of ``X`` gets the distribution the regressor predicts, e.g. a ``Normal`` whose mean and standard deviation
    both depend on the parents. The target is continuous, but the distribution can be discrete, e.g. a Poisson from a
    GLM for counts: score values with the method for the distribution's measure type, ``pmf`` rather than ``pdf`` for a
    discrete one.

    ``fit`` fits a clone of ``estimator``; skpro regressors take no sample weights. The estimator gets the parents as a
    DataFrame sorted by name, as in ``evidence_``. If the parents' names aren't all strings, they are renamed ``x0``,
    ``x1``, .... Numbers and booleans are passed as floats, and object and string parents as the categories seen in fit,
    where a value not seen in fit raises an error. A target whose name isn't a string is passed as ``str(name)``, which
    is then the column of the predicted distributions; samples keep the name. skpro matches rows by their labels, so the
    estimator is fitted by position, and for an ``X`` whose index repeats labels or is a MultiIndex, it also predicts by
    position: the predicted distribution then has ``X``'s index only if its parameters are plain numbers, as for a
    ``Normal``. A root doesn't use the estimator: every row gets the empirical distribution of ``y``, which has no
    ``log_likelihood``, as skpro's ``Empirical`` has no ``log_pmf``. Without ``X``,
    ``predict_proba`` gives the marginal distribution over the rows seen in fit: for a root, the empirical distribution
    of ``y``, and otherwise the Normal with the mean and variance of the predicted distributions as a mixture, computed
    the first time it is needed.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    estimator : skpro probabilistic regressor
        Unfitted regressor, e.g. ``skpro.regression.linear.GLMRegressor``.

    Attributes
    ----------
    estimator_ : skpro probabilistic regressor
        The fitted clone of ``estimator``, or, for a root, a fitted ``DummyProbaRegressor(strategy="empirical")``.

    Warnings
    --------
    Experimental: the API of ``pgmpy.parameterization`` may change in any release without a deprecation period.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpro.regression.linear import GLMRegressor
    >>> from pgmpy.parameterization.adapters import SkproAdapter
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame({"A": rng.normal(size=1000), "B": rng.normal(size=1000)})
    >>> y = pd.Series(1 + 2 * X["A"] - 3 * X["B"] + rng.normal(scale=0.5, size=1000), name="y")
    >>> cpd = SkproAdapter(GLMRegressor(add_constant=True)).fit(X, y)
    >>> dist = cpd.predict_proba(pd.DataFrame({"A": [0.0], "B": [0.0]}))
    >>> type(dist).__name__, float(dist.mean().iloc[0, 0].round(1)), float(np.sqrt(dist.var().iloc[0, 0]).round(1))
    ('Normal', 1.0, 0.5)
    """

    _tags = {
        "name": "skpro_adapter",
        "variable_type": ["continuous"],
        "python_dependencies": "skpro",
    }

    def __init__(self, estimator: "BaseProbaRegressor") -> None:
        self.estimator = estimator
        super().__init__()

        if (
            not hasattr(estimator, "get_tag")
            or estimator.get_tag("object_type", raise_error=False) != "regressor_proba"
        ):
            raise TypeError(f"estimator must be a skpro probabilistic regressor, but is a {type(estimator).__name__}.")

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Decide how each parent goes in: numbers and booleans as floats, and object and string columns, which
        # skpro rejects, as categories, with the categories seen here.
        features = _features(X)
        self._numbers = {name: "float64" for name, kind in preprocess_data(features)[1].items() if kind == "N"}
        self._categories = {
            name: features[name].astype("category").dtype
            for name, dtype in features.dtypes.items()
            if is_string_dtype(dtype)
        }
        # Step 2: Prepare the parents and the target, by position, as skpro matches rows by label. The target goes in as
        # floats, under a string name.
        self._training = self._prepare(X).reset_index(drop=True)
        target = y.astype(float).reset_index(drop=True)
        target = target if isinstance(self.variable_, str) else target.set_axis([str(self.variable_)], axis=1)
        self._name = target.columns[0]

        # Step 3: Fit a clone of the regressor; a root gets the empirical distribution of y. With parents, the
        # marginal is computed when first needed, as predicting every training row can be slow.
        estimator = self.estimator if self.evidence_ else DummyProbaRegressor(strategy="empirical")
        self.estimator_ = clone(estimator).fit(self._training, target)
        self._marginal = None if self.evidence_ else Empirical(spl=target.iloc[:, 0])

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the parents as skpro regressors take them: with string names, numbers and booleans as floats, and
        strings as the categories seen in fit."""
        features = _features(X)
        prepared = features.astype(self._numbers | self._categories)
        for position, name in enumerate(features.columns):
            if name in self._categories and prepared[name].isna().any():
                unseen = pd.unique(features[name][prepared[name].isna().to_numpy()]).tolist()
                raise ValueError(
                    f"X[{X.columns[position]!r}] has values not seen in fit: {unseen}. Known values: "
                    f"{self._categories[name].categories.tolist()}."
                )
        return prepared

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        # Step 1: Without X, return the marginal. With parents, it is the Normal with the mean and variance of the
        # distributions predicted for the training rows as a mixture, computed once.
        if X is None:
            if self._marginal is None:
                predicted = self.estimator_.predict_proba(self._training)
                means, variances = predicted.mean().to_numpy().ravel(), predicted.var().to_numpy().ravel()
                mean, variance = means.mean(), variances.mean() + means.var()
                if not (np.isfinite(mean) and 0 < variance < np.inf):
                    raise ValueError(
                        f"As a mixture, the distributions {type(self.estimator_).__name__} predicts for the training "
                        f"data have mean {mean:g} and variance {variance:g}, so there is no Normal marginal."
                    )
                self._marginal = Normal(mu=mean, sigma=float(np.sqrt(variance)))
            return self._marginal.clone()
        # Step 2: skpro regressors can't predict for no rows, so an empty X gets an empty Normal.
        if len(X) == 0:
            return Normal(mu=np.empty((0, 1)), sigma=1.0, index=X.index, columns=[self._name])

        # Step 3: Predict a distribution for each row. skpro matches rows by label, so repeated labels and a MultiIndex
        # are predicted by position; a distribution whose parameters are plain numbers then gets X's labels back.
        by_position = not X.index.is_unique or isinstance(X.index, pd.MultiIndex)
        features = self._prepare(X)
        dist = self.estimator_.predict_proba(features.reset_index(drop=True) if by_position else features)
        if dist.shape != (len(X), 1):
            raise ValueError(
                f"{type(self.estimator_).__name__} predicted a distribution of shape {dist.shape} for {len(X)} rows, "
                f"but SkproAdapter needs one of shape ({len(X)}, 1)."
            )
        return dist.set_params(index=X.index) if by_position and _plain(dist) else dist
