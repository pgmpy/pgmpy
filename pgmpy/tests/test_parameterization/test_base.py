import importlib
from enum import Enum
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.exceptions import NotFittedError

import pgmpy.parameterization
from pgmpy.parameterization import BaseParameter
from pgmpy.utils import ExperimentalWarning


class CountParameter(BaseParameter):
    """Records the data it was fitted on; `_predict_proba` and `_sample` return what the base passed them."""

    _tags = {"variable_type": "discrete", "supports_weighted_data": True}

    def __init__(self, pseudo_count=0):
        self.pseudo_count = pseudo_count
        super().__init__()

    def _fit(self, X, y, sample_weight):
        self.columns_ = list(X.columns)
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
        assert BaseParameter.get_class_tag("parent_type") == ["discrete", "continuous"]
        assert BaseParameter.get_class_tag("supports_weighted_data") is False
        assert BaseParameter.get_class_tag("python_dependencies") is None

    def test_fit(self, data):
        X, y = data

        # Parents are sorted by name, and _fit gets X's columns in that order.
        parameter = CountParameter().fit(X, y)
        assert parameter.is_fitted
        assert (parameter.variable_, parameter.evidence_, parameter.variable_type_) == ("t", ["a", "b"], "discrete")
        assert (parameter.columns_, parameter.n_rows_, parameter.total_weight_) == (["a", "b"], 4, 4.0)
        cyclic = CountParameter().fit(X.assign(c=X["a"])[["c", "a", "b"]], y)
        assert cyclic.evidence_ == cyclic.columns_ == ["a", "b", "c"]

        # A root has no parents: X is None or has no columns, and y can be a Series.
        for X_root in (None, pd.DataFrame(index=X.index)):
            root = CountParameter().fit(X_root, y["t"])
            assert (root.variable_, root.evidence_) == ("t", [])

        # Node names can be any hashable value. Numbers of any type sort by value, then strings, then tuples element by
        # element, then any other name by its type and repr, so names that compare equal sort the same way.
        class Color(Enum):
            RED = 1
            BLUE = 2

        names = [Color.RED, Color.BLUE, ("B", "x"), ("B", 1), "b", np.str_("a"), 2.5, np.int64(1)]
        X_named = pd.DataFrame(np.zeros((4, 8)), index=X.index, columns=pd.Index(names, tupleize_cols=False))
        renamed = CountParameter().fit(X_named, y.set_axis([("T", 0)], axis=1))
        sorted_names = [1, 2.5, "a", "b", ("B", 1), ("B", "x"), Color.BLUE, Color.RED]
        assert (renamed.variable_, renamed.evidence_) == (("T", 0), sorted_names)
        assert list(renamed.predict_proba(X_named.iloc[:, ::-1]).columns) == sorted_names

        # Parents named False and True, as from pd.get_dummies on a boolean column, are selected as columns, not rows.
        dummies = pd.get_dummies(pd.Series([True, False, True, True], index=X.index))
        boolean = CountParameter().fit(dummies, y)
        assert boolean.evidence_ == boolean.columns_ == [False, True]
        assert list(boolean.predict_proba(dummies.iloc[:, ::-1]).columns) == [False, True]

        # A failed refit leaves the object unfitted instead of half-updated.
        with pytest.raises(ValueError):
            parameter.fit(X, y.iloc[:3])
        assert not parameter.is_fitted
        assert not hasattr(parameter, "n_rows_")

        # The constructor arguments are stored, so the skbase machinery works.
        assert repr(CountParameter(pseudo_count=1)) == "CountParameter(pseudo_count=1)"
        assert CountParameter(pseudo_count=1).clone().get_params() == {"pseudo_count": 1}

        ratio = pd.array([0.0, 1.0, 1.0, 1.0], dtype="Float64")
        rejected = [
            (X, pd.concat([y, y.set_axis(["u"], axis=1)], axis=1)),  # y with two columns
            (X, y["t"].tolist()),  # y as a list
            (X.to_numpy(), y),  # X as an array
            (X, y.set_axis([0, 1, 2, 3])),  # different index
            (X.assign(t="x"), y),  # target also in X
            (X.assign(b=["u", None, "u", "v"]), y),  # None in X
            (X, y.assign(t=["0", np.nan, "1", "1"])),  # NaN in y
            (X.astype("string").assign(b=pd.array(["u", pd.NA, "u", "v"], dtype="string")), y),  # pd.NA in X
            (X.assign(c=[0.0, np.inf, 0.0, 0.0]), y),  # inf in X
            (X.assign(c=np.array([0.0, np.inf, 0.0, 0.0], dtype=np.float32)), y),  # inf in a float32 column
            (X, y.assign(t=[0.0, 1.0, -np.inf, 1.0])),  # -inf in y
            (X.assign(c=ratio / ratio), y),  # NaN from 0/0 in a Float64 column, which isna() doesn't flag
            (X.assign(c=[1j, 0, 0, 0]), y),  # complex values in X
            (X, y.assign(t=[1j, 0, 0, 0])),  # complex values in y
            (X.iloc[:0], y.iloc[:0]),  # no rows
            (pd.concat([X, X[["a"]]], axis=1), y),  # duplicate column names in X
        ]
        for X_bad, y_bad in rejected:
            with pytest.raises(ValueError):
                CountParameter().fit(X_bad, y_bad)

    def test_types(self, data):
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
        ordered = pd.Series(pd.Categorical([2, 1, 2, 2], categories=[1, 2], ordered=True), index=X.index, name="t")
        assert AnyParameter().fit(X, ordered).variable_type_ == "discrete"

        # Continuous parents must be numeric, in fit and at prediction, where discrete ones can have any labels.
        # Booleans of every kind count as numeric.
        class LinearParameter(CountParameter):
            _tags = {"variable_type": "continuous", "parent_type": "continuous"}

        numeric = pd.DataFrame({"a": [0.5, 1, 2, 3], "b": [True, False, True, True]}, index=X.index)
        linear = LinearParameter().fit(numeric, y.astype(float))
        for dtype in ["boolean", "Sparse[bool]"] + (["bool[pyarrow]"] if find_spec("pyarrow") else []):
            LinearParameter().fit(numeric.astype({"b": dtype}), y.astype(float).astype(bool).astype(dtype))
        for column in (X["a"], X["a"].astype("category"), pd.Categorical([1, 2, 1, 2]), pd.to_datetime(["2020"] * 4)):
            with pytest.raises(ValueError, match="numeric"):
                LinearParameter().fit(numeric.assign(a=column), y.astype(float))
            with pytest.raises(ValueError, match="numeric"):
                linear.predict_proba(numeric.assign(a=column))

    def test_sample_weight(self, data):
        X, y = data

        assert CountParameter().fit(X, y, sample_weight=[0.5, 1, 0, 2]).total_weight_ == 3.5

        # A Series of weights must have y's index; arrays and lists apply by position.
        weights = pd.Series([0.5, 1, 0, 2], index=y.index)
        assert CountParameter().fit(X, y, sample_weight=weights).total_weight_ == 3.5
        with pytest.raises(ValueError, match="index"):
            CountParameter().fit(X, y, sample_weight=weights.iloc[::-1])
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

        # Columns are matched by name and passed on in evidence_ order, so their order doesn't matter.
        parameter = CountParameter().fit(X, y)
        pd.testing.assert_frame_equal(parameter.predict_proba(X), X[["a", "b"]])
        ratio = pd.array([0.0, 1.0, 1.0, 1.0], dtype="Float64")
        for X_bad in (
            X[["a"]],
            X.assign(c=1),
            X.rename(columns={"a": "z"}),
            X.assign(a=["x", None, "y", "y"]),
            X.assign(a=[0.0, np.inf, 0.0, 0.0]),
            X.assign(a=ratio / ratio),
            X.assign(a=[1j, 0, 0, 0]),
        ):
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
        sampled_X, n_samples, random_state = parameter.sample(X, n_samples=2, random_state=0)
        pd.testing.assert_frame_equal(sampled_X, X[["a", "b"]])
        assert (n_samples, random_state) == (2, 0)
        assert parameter.sample(n_samples=3) == (None, 3, None)

        root = CountParameter().fit(None, y)
        assert root.sample(n_samples=0) == (None, 0, None)
        with pytest.raises(ValueError):
            root.sample()
        for n_samples in (-1, 2.5, "3", True):
            with pytest.raises(ValueError, match="n_samples"):
                root.sample(n_samples=n_samples)

    def test_sample_from_ppf(self, data):
        # skpro's sample() takes no random_state, so for its distributions the default _sample applies the ppf to
        # uniform values from numpy.random.default_rng(random_state), in the layout of NominalDistribution's samples.
        distributions = pytest.importorskip("skpro.distributions")
        from pgmpy.parameterization.distributions import NominalDistribution

        class NormalParameter(CountParameter):
            _sample = BaseParameter._sample

            def _predict_proba(self, X):
                if X is None:
                    return distributions.Normal(mu=1.0, sigma=2.0)
                return distributions.Normal(mu=np.arange(len(X))[:, None], sigma=2.0, index=X.index, columns=["t"])

        # Rows are matched by position, so repeated labels and a MultiIndex, like a parent's own draws, keep them apart.
        X, y = data
        parameter = NormalParameter().fit(X, y)
        multi = pd.MultiIndex.from_arrays([[0, 0, 1, 1], ["p", "q", "p", "q"]], names=["draw", None])
        for X_test in (X, X.iloc[[0, 0, 1, 2]], X.set_axis(multi)):
            for n_samples in (None, 1, 2, 0):
                samples = parameter.sample(X_test, n_samples=n_samples, random_state=0)
                uniform = np.random.default_rng(0).random((1 if n_samples is None else n_samples, len(X_test)))
                np.testing.assert_allclose(samples["t"], (np.arange(len(X_test)) + 2 * norm.ppf(uniform)).ravel())
                nominal = NominalDistribution(np.full((len(X_test), 2), 0.5), ["u", "v"], index=X_test.index)
                expected_index = nominal.sample(n_samples, random_state=0).index
                pd.testing.assert_index_equal(samples.index, expected_index, exact=True)
        assert parameter.sample(X, n_samples=1).index.equals(pd.MultiIndex.from_arrays([[0] * len(X), X.index]))

        # Distributions that look rows up by label, such as skpro's Empirical, still draw each row from its own values:
        # row r of X can only take the values r, r + len(X) and r + 2 * len(X).
        class EmpiricalParameter(NormalParameter):
            def _predict_proba(self, X):
                values = pd.DataFrame(
                    {"t": np.arange(3.0 * len(X))}, index=pd.MultiIndex.from_product([range(3), X.index])
                )
                return distributions.Empirical(spl=values, index=X.index, columns=["t"])

        empirical = EmpiricalParameter().fit(X, y)
        for X_test in (X.iloc[[0, 0, 1, 2]], X.set_axis(multi)):
            samples = empirical.sample(X_test, n_samples=20, random_state=0)["t"].to_numpy().reshape(20, len(X_test))
            np.testing.assert_array_equal(samples % len(X_test), np.tile(np.arange(len(X_test)), (20, 1)))

        for n_samples in (3, 0):
            samples = parameter.sample(n_samples=n_samples, random_state=0)
            expected = 1 + 2 * norm.ppf(np.random.default_rng(0).random(n_samples))
            pd.testing.assert_frame_equal(samples, pd.DataFrame({"t": expected}), rtol=1e-10)

    def test_equality(self, data):
        X, y = data

        assert CountParameter() == CountParameter()
        assert CountParameter(pseudo_count=1) != CountParameter()

        # Fitted parameters compare by what they learned, and equal ones hash equally.
        fitted, same, other = CountParameter().fit(X, y), CountParameter().fit(X, y), CountParameter().fit(X[:3], y[:3])
        assert fitted == same and hash(fitted) == hash(same)
        assert fitted != other
        assert other != CountParameter()

        # Floats compare with numpy.allclose's tolerance, as arrays do, but names compare exactly.
        assert CountParameter().fit(X, y, sample_weight=[1, 1, 1, 1 + 1e-12]) == fitted
        assert CountParameter().fit(X, y, sample_weight=[1, 1, 1, 1.1]) != fitted
        named = {name: CountParameter().fit(X, y.set_axis([name], axis=1)) for name in (1000.0, 1000.01, "t")}
        assert named[1000.0] != named["t"] and named["t"] != named[1000.0] and named[1000.0] != named[1000.01]
