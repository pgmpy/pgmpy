from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import
from sklearn.base import clone

from pgmpy.parameterization._base import BaseParameterization
from pgmpy.parameterization.adapters.sklearn import _features
from pgmpy.utils import preprocess_data

Normal = _safe_import("skpro.distributions.Normal")

if TYPE_CHECKING:
    from skpro.regression.base import BaseProbaRegressor


class SkproAdapter(BaseParameterization):
    """Parameterization from a skpro probabilistic regressor.

    Each row of ``X`` gets the distribution the regressor predicts, e.g. a ``Normal`` whose mean and standard deviation
    both depend on the parents. The target is continuous, but the distribution can be discrete, e.g. a Poisson from a
    GLM for counts: score values with the method for the distribution's measure type, ``pmf`` rather than ``pdf`` for a
    discrete one.

    ``fit`` fits a clone of ``estimator``; skpro regressors take no sample weights. The estimator gets the parents as a
    DataFrame sorted by name, as in ``evidence_``. If the parents' names aren't all strings, they are renamed ``x0``,
    ``x1``, .... Numbers and booleans are passed as floats, and other parents, e.g. strings or Categoricals, as the
    categories seen in fit, where a value not seen in fit raises an error. A target whose name isn't a string is passed
    as ``str(name)``, which is then the column of the predicted distributions; samples keep the name. skpro matches rows
    by their labels, so the estimator is fitted by position, and ``predict_proba`` raises for an ``X`` with repeated
    labels; ``predict``, ``sample`` and ``log_likelihood`` take any index. A root fits the estimator on a constant
    column, so that it is the same model without parents and can be scored like one. An estimator that adds its own
    intercept, such as ``GLMRegressor(add_constant=True)``, then has two: least squares still fits exactly, but the
    iterative fit of e.g. a Poisson GLM can end slightly off.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    estimator : skpro probabilistic regressor
        Unfitted regressor, e.g. ``skpro.regression.linear.GLMRegressor``.

    Attributes
    ----------
    estimator_ : skpro probabilistic regressor
        The fitted clone of ``estimator``; a root's is fitted on a constant column.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpro.regression.linear import GLMRegressor
    >>> from pgmpy.parameterization import SkproAdapter
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
        # Step 1: Decide how each parent goes in: numbers and booleans as floats, and the others, which skpro rejects as
        # objects or strings, as categories, with the categories seen here rather than a Categorical's declared ones.
        features = _features(X)
        kinds = preprocess_data(features)[1]
        self._numbers = {name: "float64" for name, kind in kinds.items() if kind == "N"}
        self._categories = {
            name: features[name].astype(object).astype("category").dtype for name, kind in kinds.items() if kind != "N"
        }
        # Step 2: Prepare the parents and the target, by position, as skpro matches rows by label. A root gets a
        # constant column instead, so that its estimator is the same model without parents. The target goes in as
        # floats, under a string name.
        training = self._prepare(X).reset_index(drop=True) if self.evidence_ else _constant(len(y))
        target = y.astype(float).reset_index(drop=True)
        target = target if isinstance(self.variable_, str) else target.set_axis([str(self.variable_)], axis=1)
        self._name = target.columns[0]

        # Step 3: Fit a clone of the regressor.
        self.estimator_ = clone(self.estimator).fit(training, target)

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
        # Step 1: Without X, which only a root gets, return the distribution predicted for one constant row.
        if X is None:
            return self.estimator_.predict_proba(_constant(1)).iat[0, 0]

        # Step 2: skpro regressors that wrap sklearn estimators can't predict for no rows, so an empty X gets an empty
        # Normal.
        if len(X) == 0:
            return Normal(mu=np.empty((0, 1)), sigma=1.0, index=X.index, columns=[self._name])

        # Step 3: Predict a distribution for each row, from the parents, or from the constant column for a root. skpro
        # matches rows by label, which mixes up rows with repeated labels.
        if not X.index.is_unique:
            raise ValueError(
                "SkproAdapter.predict_proba needs X with unique labels, as skpro matches rows by label: pass "
                "X.reset_index(drop=True). predict, sample and log_likelihood take any index."
            )
        features = self._prepare(X) if self.evidence_ else _constant(len(X)).set_axis(X.index)
        dist = self.estimator_.predict_proba(features)
        if dist.shape != (len(X), 1):
            raise ValueError(
                f"{type(self.estimator_).__name__} predicted a distribution of shape {dist.shape} for {len(X)} rows, "
                f"but SkproAdapter needs one of shape ({len(X)}, 1)."
            )
        return dist


def _constant(n_rows: int) -> pd.DataFrame:
    """Return the constant column that a root's estimator is fitted and predicts on, as its only feature."""
    return pd.DataFrame({"constant": np.ones(n_rows)})
