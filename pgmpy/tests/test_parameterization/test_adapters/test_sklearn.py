import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline

from pgmpy.parameterization.adapters import SklearnAdapter
from pgmpy.parameterization.cpds import TabularCPD

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


@pytest.fixture(scope="module")
def data():
    """Y = 1.25 + 2.5 sin(A) + 0.75 B^2 - 1.5 C + N(0, 0.2^2), on rows labelled with strings."""
    rng = np.random.default_rng(42)
    X, X_test = (
        pd.DataFrame(rng.uniform(-2.5, 2.5, (n, 3)), columns=["A", "B", "C"], index=[f"{name}{i}" for i in range(n)])
        for name, n in (("train", 1000), ("test", 200))
    )
    y = pd.Series(1.25 + 2.5 * np.sin(X["A"]) + 0.75 * X["B"] ** 2 - 1.5 * X["C"], name="Y")
    return X, y + rng.normal(scale=0.2, size=len(y)), X_test


class NanClassifier(ClassifierMixin, BaseEstimator):
    """Predicts equal probabilities, except NaN for the first row."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probs = np.full((len(X), len(self.classes_)), 1 / len(self.classes_))
        probs[0] = np.nan
        return probs


class TestSklearnAdapter:
    def test_tags(self):
        # The target's type comes from the estimator, as does support for sample weights: an estimator gets them if its
        # fit takes sample_weight, or any keyword argument, as meta-estimators such as GridSearchCV do.
        for estimator, variable_type, weighted in (
            (LinearRegression(), "continuous", True),
            (LogisticRegression(), "discrete", True),
            (KNeighborsClassifier(), "discrete", False),
            (GridSearchCV(LogisticRegression(), {"C": [1.0]}), "discrete", True),
        ):
            adapter = SklearnAdapter(estimator)
            assert adapter.get_tag("variable_type") == [variable_type]
            assert adapter.get_tag("supports_weighted_data") is weighted
        assert SklearnAdapter.get_class_tag("python_dependencies") == "skpro"

        # The tags follow the estimator after set_params, also a nested one.
        adapter = SklearnAdapter(Pipeline([("model", LogisticRegression())]))
        assert adapter.set_params(estimator__model=Ridge()).get_tag("variable_type") == ["continuous"]
        assert adapter.set_params(estimator=LogisticRegression()).get_tag("variable_type") == ["discrete"]

        for estimator in ("not an estimator", RidgeClassifier()):  # RidgeClassifier has no predict_proba
            with pytest.raises(TypeError):
                SklearnAdapter(estimator)

    def test_regressor(self, data):
        X, y, X_test = data
        estimator = LinearRegression()
        adapter = SklearnAdapter(estimator).fit(X, y)

        # A clone is fitted, and predict gives its predictions, with X's index.
        assert adapter.estimator_ is not estimator and not hasattr(estimator, "coef_")
        assert adapter.estimator_.feature_names_in_.tolist() == ["A", "B", "C"]
        expected = pd.DataFrame({"Y": adapter.estimator_.predict(X_test)}, index=X_test.index)
        pd.testing.assert_frame_equal(adapter.predict(X_test), expected)

        # A regressor gives point predictions, not a distribution, so it has no probabilities, samples or likelihood.
        for call in (
            lambda: adapter.predict_proba(X_test),
            lambda: adapter.sample(X_test),
            lambda: adapter.log_likelihood(X, y),
        ):
            with pytest.raises(TypeError, match="point predictions"):
                call()

        # Sample weights go to the estimator.
        weights = np.linspace(0.1, 2, len(y))
        weighted = SklearnAdapter(LinearRegression()).fit(X, y, sample_weight=weights)
        expected = LinearRegression().fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(weighted.estimator_.coef_, expected.coef_)

        # A regressor that predicts a 1-D array, like a random forest, still gives one row per row of X, and one that
        # reproduces its training data, like KNN with one neighbour, fits. Parents whose names aren't all strings, which
        # sklearn can't take as feature names, go in by position.
        forest = SklearnAdapter(RandomForestRegressor(n_estimators=10, random_state=0)).fit(X, y)
        assert forest.predict(X_test).shape == (len(X_test), 1)
        SklearnAdapter(KNeighborsRegressor(n_neighbors=1)).fit(X, y)
        renamed = SklearnAdapter(LinearRegression()).fit(X.set_axis([0, "B", ("C", 1)], axis=1), y)
        np.testing.assert_allclose(renamed.estimator_.coef_, adapter.estimator_.coef_)

        # The estimator gets a copy of y, so a regressor that keeps a view of it, like KNN, doesn't change when the
        # caller edits their data.
        target = y.copy()
        knn = SklearnAdapter(KNeighborsRegressor(n_neighbors=2)).fit(X, target)
        predictions = knn.predict(X_test)
        target.iloc[:] = 0.0
        pd.testing.assert_frame_equal(knn.predict(X_test), predictions)

        # A root doesn't use the estimator: it predicts the weighted mean of y.
        root = SklearnAdapter(LinearRegression()).fit(None, y, sample_weight=weights)
        assert isinstance(root.estimator_, DummyRegressor)
        np.testing.assert_allclose(root.predict(pd.DataFrame(index=[7, 8]))["Y"], [np.average(y, weights=weights)] * 2)
        assert adapter.predict(X_test.iloc[:0]).shape == root.predict(pd.DataFrame(index=[])).shape == (0, 1)

        # A prediction that isn't finite, e.g. from isotonic regression outside its training range, raises.
        isotonic = SklearnAdapter(IsotonicRegression()).fit(X[["A"]], y)
        with pytest.raises(ValueError, match="IsotonicRegression.*'out'"):
            isotonic.predict(pd.DataFrame({"A": [0.0, 5.0]}, index=["in", "out"]))

        # A fitted estimator compares by identity, so a fitted adapter only equals itself.
        assert adapter == adapter and len({adapter, adapter}) == 1
        assert adapter != SklearnAdapter(LinearRegression()).fit(X, y)

    def test_classifier(self, data):
        X, y, X_test = data
        labels = pd.Series(np.where(y > y.median(), "hi", "lo"), index=y.index, name="Y")
        adapter = SklearnAdapter(LogisticRegression()).fit(X, labels)

        # The classifier's probabilities for each row, over the original labels.
        dist = adapter.predict_proba(X_test)
        assert (dist.index.tolist(), dist.columns.tolist()) == (X_test.index.tolist(), ["Y"])
        assert list(dist.categories) == list(adapter.classes_) == ["hi", "lo"]
        np.testing.assert_allclose(np.asarray(dist.probs), adapter.estimator_.predict_proba(X_test))

        # Labels are sorted by value, as in TabularCPD, also for a Categorical that orders them otherwise.
        reordered = labels.astype(pd.CategoricalDtype(["mid", "lo", "hi"]))
        assert SklearnAdapter(LogisticRegression()).fit(X, reordered).classes_.tolist() == ["hi", "lo"]
        rows = np.arange(len(X)), (labels == "lo").to_numpy().astype(int)
        np.testing.assert_allclose(
            adapter.log_likelihood(X, labels)["Y"], np.log(adapter.estimator_.predict_proba(X)[rows])
        )
        most_probable = adapter.classes_[adapter.estimator_.predict_proba(X_test).argmax(axis=1)]
        pd.testing.assert_frame_equal(adapter.predict(X_test), pd.DataFrame({"Y": most_probable}, index=X_test.index))
        samples = adapter.sample(X_test, n_samples=20, random_state=0)["Y"]
        assert samples.isin(["hi", "lo"]).all() and not samples.equals(adapter.sample(X_test, 20, random_state=1)["Y"])

        # The labels are encoded before fitting, so the estimator can't change them: nullable booleans stay booleans,
        # and a TabularCPD child fitted on the original data takes the samples.
        flag = (X["A"] > 0).astype("boolean").rename("F")
        boolean = SklearnAdapter(LogisticRegression()).fit(X[["B"]], flag)
        assert list(boolean.predict_proba(X_test[["B"]]).categories) == [False, True]
        child = TabularCPD().fit(flag.to_frame(), labels)
        child.predict_proba(boolean.sample(X_test[["B"]], random_state=0))

        # A single observed class gets all the probability without the estimator, which some can't predict with.
        single = SklearnAdapter(HistGradientBoostingClassifier()).fit(X, pd.Series("no", index=X.index, name="Y"))
        assert isinstance(single.estimator_, DummyClassifier)
        assert (single.sample(X_test)["Y"] == "no").all()

        # A root gets the weighted frequencies of the labels.
        weights = np.where(labels == "hi", 3.0, 1.0)
        root = SklearnAdapter(LogisticRegression()).fit(None, labels, sample_weight=weights)
        share = weights[labels == "hi"].sum() / weights.sum()
        np.testing.assert_allclose(np.asarray(root.predict_proba().probs), [share, 1 - share])

        # Probabilities that aren't a distribution, e.g. NaN, are rejected by the NominalDistribution built from them.
        with pytest.raises(ValueError, match="must sum to 1"):
            SklearnAdapter(NanClassifier()).fit(X, labels).predict_proba(X)
