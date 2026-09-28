import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import pgmpy.parameterization
from pgmpy.parameterization import BaseParameter
from pgmpy.utils import ExperimentalWarning


class CountParameter(BaseParameter):
    """Records how much data it was fitted on; `_predict_proba` and `_sample` return what the base passed them."""

    _tags = {"variable_type": "discrete", "supports_weighted_data": True}

    def __init__(self, pseudo_count=0):
        self.pseudo_count = pseudo_count
        super().__init__()

    def _fit(self, X, y, sample_weight):
        self.n_rows_ = len(y)
        self.total_weight_ = float(len(y) if sample_weight is None else sample_weight.sum())

    def _predict_proba(self, X):
        return X

    def _sample(self, X, n_samples, random_state):
        return X, n_samples, random_state


@pytest.fixture
def data():
    X = pd.DataFrame({"b": ["u", "v", "u", "v"], "a": ["x", "x", "y", "y"]}, index=[10, 11, 12, 13])
    y = pd.DataFrame({"t": ["0", "1", "1", "1"]}, index=X.index)
    return X, y


class TestBaseParameter:
    def test_experimental_warning(self):
        with pytest.warns(ExperimentalWarning, match="pgmpy.parameterization is experimental"):
            importlib.reload(pgmpy.parameterization)

    def test_tags(self):
        assert BaseParameter.get_class_tag("object_type") == "parameterization"
        assert BaseParameter.get_class_tag("variable_type") == ["discrete", "continuous"]
        assert BaseParameter.get_class_tag("supports_weighted_data") is False
        assert BaseParameter.get_class_tag("python_dependencies") is None
        for tag in ("local:plug_in", "global:plug_in", "local:full_bayesian", "global:full_bayesian"):
            assert BaseParameter.get_class_tag(tag) == []

    def test_fit(self, data):
        X, y = data

        parameter = CountParameter().fit(X, y)
        assert parameter.is_fitted
        assert (parameter.variable_, parameter.evidence_, parameter.variable_type_) == ("t", ["b", "a"], "discrete")
        assert (parameter.n_rows_, parameter.total_weight_) == (4, 4.0)

        # A root has no parents: X is None or has no columns, and y can be a Series.
        for X_root in (None, pd.DataFrame(index=X.index)):
            root = CountParameter().fit(X_root, y["t"])
            assert (root.variable_, root.evidence_) == ("t", [])

        # Node names can be any hashable value.
        renamed = CountParameter().fit(X.set_axis([0, ("B", 1)], axis=1), y.set_axis([("T", 0)], axis=1))
        assert (renamed.variable_, renamed.evidence_) == (("T", 0), [0, ("B", 1)])

        # A failed refit leaves the object unfitted instead of half-updated.
        with pytest.raises(ValueError):
            parameter.fit(X, y.iloc[:3])
        assert not parameter.is_fitted
        assert not hasattr(parameter, "n_rows_")

        # The constructor arguments are stored, so the skbase machinery works.
        assert repr(CountParameter(pseudo_count=1)) == "CountParameter(pseudo_count=1)"
        assert CountParameter(pseudo_count=1).clone().get_params() == {"pseudo_count": 1}

        rejected = [
            (X, pd.concat([y, y.set_axis(["u"], axis=1)], axis=1)),  # y with two columns
            (X, y["t"].tolist()),  # y as a list
            (X.to_numpy(), y),  # X as an array
            (X, y.set_axis([0, 1, 2, 3])),  # different index
            (X.assign(t="x"), y),  # target also in X
            (X.assign(b=["u", None, "u", "v"]), y),  # None in X
            (X, y.assign(t=["0", np.nan, "1", "1"])),  # NaN in y
            (X.astype("string").assign(b=pd.array(["u", pd.NA, "u", "v"], dtype="string")), y),  # pd.NA in X
            (X.iloc[:0], y.iloc[:0]),  # no rows
        ]
        for X_bad, y_bad in rejected:
            with pytest.raises(ValueError):
                CountParameter().fit(X_bad, y_bad)

    def test_variable_type(self, data):
        X, y = data

        # A discrete-only class accepts any labels, such as the integer, boolean or categorical data from simulate().
        for labels in ([0, 1, 1, 1], [True, False, False, True], pd.Categorical([2, 1, 2, 2]), [1, "b", ("c", 3), 1]):
            fitted = CountParameter().fit(X, pd.Series(labels, index=X.index, name="t"))
            assert fitted.variable_type_ == "discrete"

        class ContinuousParameter(CountParameter):
            _tags = {"variable_type": "continuous"}

        assert ContinuousParameter().fit(X, y.astype(float)).variable_type_ == "continuous"
        with pytest.raises(ValueError, match="numeric"):
            ContinuousParameter().fit(X, y)

        # A class that supports both types decides from the data.
        class AnyParameter(CountParameter):
            _tags = {"variable_type": ["discrete", "continuous"]}

        assert AnyParameter().fit(X, y.astype(float) + 0.5).variable_type_ == "continuous"
        assert AnyParameter().fit(X, y).variable_type_ == "discrete"

    def test_sample_weight(self, data):
        X, y = data

        assert CountParameter().fit(X, y, sample_weight=[0.5, 1, 0, 2]).total_weight_ == 3.5
        for weights in (
            [1, 1, 1],
            [[1], [1], [1], [1]],
            [1, -1, 1, 1],
            [1, np.nan, 1, 1],
            [1, np.inf, 1, 1],
            [0, 0, 0, 0],
        ):
            with pytest.raises(ValueError, match="sample_weight"):
                CountParameter().fit(X, y, sample_weight=weights)

        class UnweightedParameter(CountParameter):
            _tags = {"supports_weighted_data": False}

        with pytest.raises(ValueError, match="sample_weight"):
            UnweightedParameter().fit(X, y, sample_weight=[1, 1, 1, 1])

    def test_predict_proba(self, data):
        X, y = data
        with pytest.raises(NotFittedError):
            CountParameter().predict_proba(X)

        # Columns are matched by name, so their order doesn't matter.
        parameter = CountParameter().fit(X, y)
        pd.testing.assert_frame_equal(parameter.predict_proba(X[["a", "b"]]), X)
        for X_bad in (X[["a"]], X.assign(c=1), X.rename(columns={"a": "z"}), X.assign(a=["x", None, "y", "y"])):
            with pytest.raises(ValueError):
                parameter.predict_proba(X_bad)
        with pytest.raises(ValueError):
            parameter.predict_proba(X.to_numpy())

        # Without X, the subclass gives the marginal distribution, for a root or any other variable. A root also takes a
        # DataFrame without columns, whose index sets the rows.
        assert parameter.predict_proba() is None
        root = CountParameter().fit(None, y)
        assert root.predict_proba(pd.DataFrame(index=[5, 6])).index.tolist() == [5, 6]
        assert root.predict_proba() is None

    def test_sample(self, data):
        X, y = data
        with pytest.raises(NotFittedError):
            CountParameter().sample(X)

        parameter = CountParameter().fit(X, y)
        sampled_X, n_samples, random_state = parameter.sample(X[["a", "b"]], n_samples=2, random_state=0)
        pd.testing.assert_frame_equal(sampled_X, X)
        assert (n_samples, random_state) == (2, 0)
        assert parameter.sample(n_samples=3) == (None, 3, None)

        root = CountParameter().fit(None, y)
        assert root.sample(n_samples=0) == (None, 0, None)
        with pytest.raises(ValueError):
            root.sample()
        for n_samples in (-1, 2.5, "3", True):
            with pytest.raises(ValueError, match="n_samples"):
                root.sample(n_samples=n_samples)

    def test_equality(self, data):
        X, y = data

        assert CountParameter() == CountParameter()
        assert CountParameter(pseudo_count=1) != CountParameter()

        # Fitted parameters compare by what they learned, and equal ones hash equally.
        fitted, same, other = CountParameter().fit(X, y), CountParameter().fit(X, y), CountParameter().fit(X[:3], y[:3])
        assert fitted == same and hash(fitted) == hash(same)
        assert fitted != other
        assert other != CountParameter()
