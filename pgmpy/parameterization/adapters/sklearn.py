import inspect
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from pgmpy.parameterization._base import BaseParameter
from pgmpy.parameterization.distributions import NominalDistribution


class SklearnAdapter(BaseParameter):
    """Parameterization from a scikit-learn classifier or regressor.

    A classifier models a discrete target: each row of ``X`` gets a ``NominalDistribution`` with the classifier's
    probabilities of the target's labels. A regressor gives point predictions only: ``predict`` returns them, while
    ``predict_proba``, ``sample`` and ``log_likelihood`` raise a ``NotImplementedError``, as a regressor has no
    distribution, and its ``capability:distribution`` tag is False. For a distribution, use ``SkproAdapter`` with a
    skpro probabilistic regressor.

    ``fit`` fits a clone of ``estimator``, with the sample weights if given. The estimator gets the parents as a
    DataFrame sorted by name, as in ``evidence_``, so a pipeline must select its columns by name, not position. If the
    parents' names aren't all strings, they are renamed ``x0``, ``x1``, .... A root, or a classifier that sees a single
    label, doesn't use the estimator: it gets the weighted mean of ``y``, or the weighted frequencies of its labels.

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
        For a classifier, the target's labels, sorted, in the order of the predicted probabilities.

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
    >>> cpd.predict(pd.DataFrame({"temp": [20.0], "rain": ["yes"]})).round(1)["sales"].tolist()
    [42.1]

    A classifier gives the probability of each label:

    >>> busy = pd.Series(np.where(y > 45, "busy", "quiet"), name="day")
    >>> classifier = SklearnAdapter(make_pipeline(encode_rain, LogisticRegression())).fit(X, busy)
    >>> classifier.classes_.tolist()
    ['busy', 'quiet']
    """

    _tags = {
        "name": "sklearn_adapter",
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
            self.set_tags(variable_type=["discrete"])
        elif is_regressor(estimator):
            self.set_tags(**{"variable_type": ["continuous"], "capability:distribution": False})
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

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Prepare the target, and the parents under names sklearn takes.
        target = y.iloc[:, 0]
        fit_params = {} if sample_weight is None else {"sample_weight": sample_weight}
        features = _features(X)

        # Step 2: Fit a classifier on the labels, encoded first, as estimators may change them in classes_, e.g.
        # nullable booleans to floats. Labels are sorted by value, as in TabularCPD, so a Categorical goes in as its
        # values: pd.factorize would order it by category.
        if self.variable_type_ == "discrete":
            codes, labels = pd.factorize(target.to_numpy(), sort=True)
            estimator = self.estimator if self.evidence_ and len(labels) > 1 else DummyClassifier(strategy="prior")
            self.estimator_ = clone(estimator).fit(features, codes, **fit_params)
            self.classes_ = np.asarray(labels)[self.estimator_.classes_]
            return

        # Step 3: Fit a regressor on a copy of y, so that one that keeps a view of y, such as KNeighborsRegressor,
        # doesn't change when the caller edits their data.
        estimator = self.estimator if self.evidence_ else DummyRegressor()
        self.estimator_ = clone(estimator).fit(features, target.copy(), **fit_params)

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        if self.variable_type_ == "continuous":
            raise NotImplementedError(
                f"{type(self.estimator).__name__} is a regressor, which gives point predictions, not a distribution: "
                "use predict, or SkproAdapter with a skpro probabilistic regressor."
            )
        # Only a root gets X=None, and its DummyClassifier holds the weighted frequencies of the labels.
        if X is None:
            return NominalDistribution(probs=self.estimator_.class_prior_.copy(), categories=list(self.classes_))
        # sklearn estimators can't predict for no rows.
        probs = np.empty((0, len(self.classes_))) if len(X) == 0 else self.estimator_.predict_proba(_features(X))
        return NominalDistribution(probs=probs, categories=list(self.classes_), index=X.index, columns=[self.variable_])

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.variable_type_ == "discrete":
            return super()._predict(X)
        if len(X) == 0:
            return pd.DataFrame(np.empty((0, 1)), index=X.index, columns=[self.variable_])
        # Some regressors predict a column, others a 1-D array.
        predictions = np.asarray(self.estimator_.predict(_features(X)), dtype=float).reshape(-1, 1)
        invalid = ~np.isfinite(predictions[:, 0])
        if invalid.any():
            raise ValueError(
                f"{type(self.estimator_).__name__} predicted values that aren't finite for rows "
                f"{X.index[invalid].tolist()}, e.g. outside the range of its training data."
            )
        return pd.DataFrame(predictions, index=X.index, columns=[self.variable_])


def _features(X: pd.DataFrame) -> pd.DataFrame:
    """Return the parents under names an estimator takes: sklearn only keeps feature names that are all strings."""
    if all(isinstance(name, str) for name in X.columns):
        return X
    return X.set_axis([f"x{position}" for position in range(X.shape[1])], axis=1)
