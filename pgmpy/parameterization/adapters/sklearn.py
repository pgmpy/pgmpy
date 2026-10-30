import inspect
from typing import Any

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from pgmpy.parameterization._base import BaseParameter
from pgmpy.parameterization.distributions import NominalDistribution

Normal = _safe_import("skpro.distributions.Normal")


class SklearnAdapter(BaseParameter):
    """Parameterization from a scikit-learn classifier or regressor.

    A classifier models a discrete target: each row of ``X`` gets a ``NominalDistribution`` with the classifier's
    probabilities of the target's labels. A regressor models a continuous target with Gaussian noise of constant
    variance: each row gets a skpro ``Normal`` with the regressor's prediction as mean and, as standard deviation, the
    root mean square of the (weighted) residuals on the training data. So a flexible model that fits its training data
    closely, such as a random forest, gets distributions that are too narrow. For noise that changes with the parents,
    use ``SkproAdapter`` with a skpro regressor that models it.

    ``fit`` fits a clone of ``estimator``, with the sample weights if given. The estimator gets the parents as a
    DataFrame sorted by name, as in ``evidence_``, so a pipeline must select its columns by name, not position. If the
    parents' names aren't all strings, they are renamed ``x0``, ``x1``, .... A root, or a classifier that sees a single
    label, doesn't use the estimator. Without ``X``, ``predict_proba`` gives the marginal distribution over the rows
    seen in fit: for a classifier, the average predicted probabilities, and for a regressor, the Normal with the mean
    and variance of the predicted Normals as a mixture.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    estimator : sklearn classifier or regressor
        Unfitted estimator. A classifier must implement ``predict_proba``.

    Attributes
    ----------
    estimator_ : sklearn estimator
        The fitted clone of ``estimator``, or, for a root and for a classifier that saw a single label, a fitted
        ``DummyRegressor`` or ``DummyClassifier``.
    classes_ : numpy.ndarray
        For a classifier, the target's labels, in the order of the predicted probabilities.
    std_ : float
        For a regressor, the standard deviation of the noise.

    Warnings
    --------
    Experimental: the API of ``pgmpy.parameterization`` may change in any release without a deprecation period.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.compose import make_column_transformer
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from pgmpy.parameterization.adapters import SklearnAdapter
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame({"temp": rng.normal(20, 5, size=500), "rain": rng.choice(["no", "yes"], size=500)})
    >>> y = pd.Series(10 + 2 * X["temp"] - 8 * (X["rain"] == "yes") + rng.normal(size=500), name="sales")

    The pipeline selects the column to encode by name:

    >>> encode_rain = make_column_transformer((OneHotEncoder(drop="first"), ["rain"]), remainder="passthrough")
    >>> cpd = SklearnAdapter(make_pipeline(encode_rain, LinearRegression())).fit(X, y)
    >>> round(cpd.std_, 1)
    1.0
    >>> cpd.predict_proba(pd.DataFrame({"temp": [20.0], "rain": ["yes"]})).mean().round(1)["sales"].tolist()
    [42.1]

    A classifier gives the probability of each label:

    >>> busy = pd.Series(np.where(y > 45, "busy", "quiet"), name="day")
    >>> classifier = SklearnAdapter(make_pipeline(encode_rain, LogisticRegression())).fit(X, busy)
    >>> classifier.classes_.tolist()
    ['busy', 'quiet']
    """

    _tags = {
        "variable_type": ["discrete", "continuous"],
        "python_dependencies": "skpro",
    }

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator
        super().__init__()

        if not isinstance(estimator, BaseEstimator):
            raise TypeError(f"estimator must be a scikit-learn estimator, but is a {type(estimator).__name__}.")
        if is_classifier(estimator):
            if not callable(getattr(estimator, "predict_proba", None)):
                raise TypeError(f"A classifier must implement predict_proba, but {type(estimator).__name__} doesn't.")
            self.set_tags(variable_type="discrete")
        elif is_regressor(estimator):
            self.set_tags(variable_type="continuous")
        else:
            raise TypeError(
                f"estimator must be a scikit-learn classifier or regressor, but is a {type(estimator).__name__}."
            )
        # Meta-estimators such as GridSearchCV pass sample_weight on through **kwargs.
        parameters = inspect.signature(estimator.fit).parameters.values()
        weighted = any(
            parameter.name == "sample_weight" or parameter.kind is parameter.VAR_KEYWORD for parameter in parameters
        )
        self.set_tags(supports_weighted_data=weighted)

    def set_params(self, **params: Any) -> "SklearnAdapter":
        """Set the parameters, and take the tags from the updated estimator."""
        # skbase resets before it sets nested parameters, such as estimator__model, so reset again for the tags.
        return super().set_params(**params).reset()

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        target = y.iloc[:, 0]
        weights = np.ones(len(target)) if sample_weight is None else sample_weight
        fit_params = {} if sample_weight is None else {"sample_weight": sample_weight}
        features = _features(X)

        if self.variable_type_ == "discrete":
            # The labels are encoded first, as estimators may change them in classes_, e.g. nullable booleans to floats.
            codes, labels = pd.factorize(target, sort=True)
            estimator = self.estimator if self.evidence_ and len(labels) > 1 else DummyClassifier(strategy="prior")
            self.estimator_ = clone(estimator).fit(features, codes, **fit_params)
            self.classes_ = np.asarray(labels)[self.estimator_.classes_]
            self._marginal = np.average(self._probabilities(features), axis=0, weights=weights)
            return

        estimator = self.estimator if self.evidence_ else DummyRegressor()
        self.estimator_ = clone(estimator).fit(features, target, **fit_params)
        # Rows without weight don't count, and could overflow the sums of squares.
        positive = weights > 0
        predictions, weights = self._predictions(features)[positive], weights[positive]
        residuals = target.to_numpy(dtype=float)[positive] - predictions
        self.std_ = float(np.sqrt(np.average(residuals**2, weights=weights)))
        if not 0 < self.std_ < np.inf:
            raise ValueError(
                f"The residuals of {type(self.estimator_).__name__} on its training data have a standard deviation of "
                f"{self.std_:g}, but the Normal needs a positive, finite one. A model that reproduces its training "
                "data, such as KNN with one neighbour, gives 0."
            )
        mean = np.average(predictions, weights=weights)
        variance = np.average((predictions - mean) ** 2, weights=weights) + self.std_**2
        self._marginal = (mean, float(np.sqrt(variance)))

    def _predictions(self, features: pd.DataFrame) -> np.ndarray:
        if len(features) == 0:
            return np.empty(0)
        # Some regressors predict a column, others a 1-D array.
        predictions = np.asarray(self.estimator_.predict(features), dtype=float).reshape(-1)
        invalid = ~np.isfinite(predictions)
        if invalid.any():
            raise ValueError(
                f"{type(self.estimator_).__name__} predicted values that aren't finite for rows "
                f"{features.index[invalid].tolist()}, e.g. outside the range of its training data."
            )
        return predictions

    def _probabilities(self, features: pd.DataFrame) -> np.ndarray:
        if len(features) == 0:
            return np.empty((0, len(self.classes_)))
        probs = self.estimator_.predict_proba(features)
        invalid = ~(np.isfinite(probs).all(axis=1) & np.isclose(probs.sum(axis=1), 1, atol=0.01))
        if invalid.any():
            raise ValueError(
                f"{type(self.estimator_).__name__} predicted probabilities that aren't a distribution for rows "
                f"{features.index[invalid].tolist()}."
            )
        return probs

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        if self.variable_type_ == "discrete":
            if X is None:
                return NominalDistribution(probs=self._marginal.copy(), categories=list(self.classes_))
            return NominalDistribution(
                probs=self._probabilities(_features(X)),
                categories=list(self.classes_),
                index=X.index,
                columns=[self.variable_],
            )
        if X is None:
            return Normal(mu=self._marginal[0], sigma=self._marginal[1])
        mu = self._predictions(_features(X)).reshape(-1, 1)
        return Normal(mu=mu, sigma=self.std_, index=X.index, columns=[self.variable_])


def _features(X: pd.DataFrame) -> pd.DataFrame:
    """Return the parents under names an estimator takes: sklearn only keeps feature names that are all strings."""
    if all(isinstance(name, str) for name in X.columns):
        return X
    return X.set_axis([f"x{position}" for position in range(X.shape[1])], axis=1)
