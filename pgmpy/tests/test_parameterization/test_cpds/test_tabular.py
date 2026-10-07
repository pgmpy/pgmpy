import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.factors.discrete import TabularCPD as LegacyTabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.parameterization.cpds import TabularCPD

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


@pytest.fixture
def discrete_data():
    """
    Counts of y for each parent combination; the MLE table divides each column by its total.

    +---------+-----------------------+-----------------------+-----------------------+
    | x1      |           0           |           1           |           2           |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | x2      |     0     |     1     |     0     |     1     |     0     |     1     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 0   |     7     |     6     |    11     |     9     |    11     |    12     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    | y = 1   |     7     |     7     |    10     |     3     |     5     |    12     |
    +---------+-----------+-----------+-----------+-----------+-----------+-----------+
    """
    rng = np.random.default_rng(seed=42)
    n_samples = 100
    X = pd.DataFrame({"x1": rng.integers(0, 3, size=n_samples), "x2": rng.integers(0, 2, size=n_samples)}).astype(str)
    y = pd.DataFrame({"y": rng.integers(0, 2, size=n_samples)}).astype(str)
    return X, y


EXPECTED_CPT = np.array(
    [[7 / 14, 6 / 13, 11 / 21, 9 / 12, 11 / 16, 12 / 24], [7 / 14, 7 / 13, 10 / 21, 3 / 12, 5 / 16, 12 / 24]]
)


class TestTabularCPD:
    def test_tags(self):
        assert TabularCPD.get_class_tag("object_type") == "parameterization"
        assert TabularCPD.get_class_tag("variable_type") == ["discrete"]
        assert TabularCPD.get_class_tag("parent_data_types") == ["discrete"]
        assert TabularCPD.get_class_tag("capability:factor") is True
        assert TabularCPD.get_class_tag("capability:exact_inference") is True
        assert TabularCPD.get_class_tag("supports_weighted_data") is True
        assert TabularCPD.get_class_tag("python_dependencies") == "skpro"

    def test_fit(self, discrete_data):
        X, y = discrete_data

        # Case 1: root node
        root = TabularCPD().fit(None, y)
        np.testing.assert_allclose(root.CPT_, [[0.56], [0.44]])
        assert (root.variable_, root.evidence_, root.state_names_) == ("y", [], {"y": ["0", "1"]})

        # Case 2: root node with sample weights that balance the two states
        weights = np.where(y["y"] == "0", 0.44, 0.56)
        np.testing.assert_allclose(TabularCPD().fit(None, y, sample_weight=weights).CPT_, [[0.5], [0.5]])

        # Case 3: conditional distribution
        cpd = TabularCPD().fit(X, y)
        np.testing.assert_allclose(cpd.CPT_, EXPECTED_CPT)
        assert cpd.state_names_ == {"y": ["0", "1"], "x1": ["0", "1", "2"], "x2": ["0", "1"]}

        # Case 4: conditional distribution with sample weights that balance every column
        weights = 1 / X.join(y).groupby(["x1", "x2", "y"])["y"].transform("size")
        np.testing.assert_allclose(TabularCPD().fit(X, y, sample_weight=weights).CPT_, np.full((2, 6), 0.5))

        # Rows pair up by position, so repeated index labels, e.g. from a bootstrap resample, don't mix them up.
        repeated = TabularCPD().fit(X.set_axis([0] * len(X)), y.set_axis([0] * len(y)), sample_weight=np.ones(len(y)))
        np.testing.assert_allclose(repeated.CPT_, EXPECTED_CPT)

    def test_matches_discrete_mle(self):
        # A random table over three parents shows any column mix-up. Rows with (x1, x2) = (2, 1) are dropped so that
        # combination is never seen, and zero weights on (x1, x3) = (0, 1) leave another combination without data.
        rng = np.random.default_rng(0)
        model = DiscreteBayesianNetwork([("x1", "y"), ("x2", "y"), ("x3", "y")])
        model.add_cpds(
            LegacyTabularCPD("x1", 3, [[0.3], [0.3], [0.4]]),
            LegacyTabularCPD("x2", 2, [[0.5], [0.5]]),
            LegacyTabularCPD("x3", 2, [[0.6], [0.4]]),
            LegacyTabularCPD("y", 3, rng.dirichlet([1, 1, 1], size=12).T, ["x1", "x2", "x3"], [3, 2, 2]),
        )
        data = model.simulate(3000, seed=0, show_progress=False)
        data = data[~((data["x1"] == 2) & (data["x2"] == 1))]
        weights = np.where((data["x1"] == 0) & (data["x3"] == 1), 0.0, rng.uniform(0.5, 2, len(data)))

        for sample_weight in (None, weights):
            expected = {cpd.variable: cpd for cpd in DiscreteMLE().fit(model, data, sample_weight).parameters_}
            for node in model.nodes:
                parents = list(model.get_parents(node))[::-1]
                cpd = TabularCPD().fit(data[parents] if parents else None, data[[node]], sample_weight=sample_weight)
                np.testing.assert_allclose(cpd.CPT_, expected[node].get_values())
                assert cpd.evidence_ == expected[node].variables[1:]
                assert cpd.state_names_ == expected[node].state_names

    def test_state_names(self, discrete_data):
        X, y = discrete_data

        # One dict covers the target and its parents, as in DiscreteMLE. Other variables are ignored, and listed states
        # never seen in the data get zero probability, or a uniform column for an unseen parent state.
        state_names = {"y": ["1", "0", "2"], "x1": ["0", "1", "2"], "x2": ["0", "1", "3"], "other": ["a"]}
        cpd = TabularCPD(state_names=state_names).fit(X, y)
        assert cpd.state_names_ == {"y": ["1", "0", "2"], "x1": ["0", "1", "2"], "x2": ["0", "1", "3"]}
        seen = [0, 1, 3, 4, 6, 7]
        np.testing.assert_allclose(cpd.CPT_[[1, 0]][:, seen], EXPECTED_CPT)
        np.testing.assert_allclose(cpd.CPT_[2, seen], 0)
        np.testing.assert_allclose(cpd.CPT_[:, [2, 5, 8]], 1 / 3)

        # The states are copied at fit.
        state_names["y"].append("9")
        assert cpd.state_names_["y"] == ["1", "0", "2"]

        with pytest.raises(ValueError, match="unexpected states"):
            TabularCPD(state_names={"y": ["0"]}).fit(X, y)
        with pytest.raises(ValueError, match="Repeated state names"):
            TabularCPD(state_names={"y": ["0", "1", "1"]}).fit(X, y)
        with pytest.raises(TypeError, match="dict"):
            TabularCPD(state_names=["0", "1"]).fit(X, y)
        assert TabularCPD(state_names={"y": iter(["0", "1"])}).fit(X, y).state_names_["y"] == ["0", "1"]

        # States that equal the data only under Python's == (here 0/1 for booleans) are caught instead of miscounted.
        with pytest.raises(ValueError, match="don't match"):
            TabularCPD(state_names={"y": [0, 1]}).fit(None, pd.Series([True, True, False], name="y"))

        # Huge but finite weights don't overflow, and an index named like a parent doesn't clash with it.
        np.testing.assert_allclose(TabularCPD().fit(X, y, sample_weight=np.full(len(y), 1e307)).CPT_, EXPECTED_CPT)
        np.testing.assert_allclose(TabularCPD().fit(X.rename_axis("x1"), y.rename_axis("x1")).CPT_, EXPECTED_CPT)

    def test_predict_proba(self, discrete_data):
        X, y = discrete_data
        cpd = TabularCPD().fit(X, y)

        dist = cpd.predict_proba(X[:5])
        expected = [[6 / 13, 7 / 13], [11 / 16, 5 / 16], [3 / 4, 1 / 4], [11 / 21, 10 / 21], [3 / 4, 1 / 4]]
        np.testing.assert_allclose(np.asarray(dist.probs), expected)
        assert (list(dist.categories), list(dist.columns)) == (["0", "1"], ["y"])

        # Predictions keep X's index, so scoring matches each row with its own prediction.
        dist = cpd.predict_proba(X.iloc[[2, 0]].set_axis(["b", "a"]))
        assert dist.index.tolist() == ["b", "a"]
        scores = dist.pmf(pd.DataFrame({"y": ["0", "1"]}, index=["a", "b"]))
        np.testing.assert_allclose(scores.loc[["a", "b"], "y"], [6 / 13, 1 / 4])

        # Parent values never seen in fit raise, including a different dtype.
        for X_unknown in (X.assign(x1="9"), X.astype(int)):
            with pytest.raises(ValueError, match="not seen in fit"):
                cpd.predict_proba(X_unknown)

        # Categorical and boolean parents are looked up by value.
        X_categorical = X.assign(x1=X["x1"].astype(pd.CategoricalDtype(["2", "0", "1"])))
        categorical = TabularCPD().fit(X_categorical, y)
        np.testing.assert_allclose(categorical.CPT_, EXPECTED_CPT)
        np.testing.assert_allclose(np.asarray(categorical.predict_proba(X_categorical[:5]).probs), expected)
        X_boolean = pd.DataFrame({"x": pd.array(X["x2"] == "1", dtype="boolean")})
        boolean = TabularCPD().fit(X_boolean, y)
        columns = X_boolean["x"][:5].astype(int).to_numpy()
        np.testing.assert_allclose(np.asarray(boolean.predict_proba(X_boolean[:5]).probs), boolean.CPT_[:, columns].T)

        # A root gives its own distribution for every row, or as a single distribution without X.
        root = TabularCPD().fit(None, y)
        rows = root.predict_proba(pd.DataFrame(index=[7, 8]))
        np.testing.assert_allclose(np.asarray(rows.probs), [[0.56, 0.44], [0.56, 0.44]])
        assert rows.index.tolist() == [7, 8]
        dist = root.predict_proba()
        assert dist.shape == () and dist.pmf("0") == pytest.approx(0.56)

        # Returned distributions hold copies, so changing them leaves the CPD as it was.
        dist.probs[:] = 0.5
        dist.categories.reverse()
        assert root.predict_proba().pmf("0") == pytest.approx(0.56) and root.state_names_["y"] == ["0", "1"]

    def test_predict(self, discrete_data):
        # The most probable state of each row, with X's index. Ties go to the first state, and the states keep their
        # dtype.
        X, y = discrete_data
        cpd = TabularCPD().fit(X, y)
        expected = pd.DataFrame({"y": ["1", "0", "0", "0", "0"]}, index=list("abcde"))
        pd.testing.assert_frame_equal(cpd.predict(X[:5].set_axis(list("abcde"))), expected)
        tied = TabularCPD.from_values("y", 2, [[0.5], [0.5]], state_names={"y": [1, 0]})
        pd.testing.assert_frame_equal(
            tied.predict(pd.DataFrame(index=[7, 8])), pd.DataFrame({"y": [1, 1]}, index=[7, 8])
        )

    def test_log_likelihood(self, discrete_data):
        # The log of each row's probability in the table, with X's index, also when labels repeat. A state the table
        # doesn't have is impossible, and a root is scored with X=None.
        X, y = discrete_data
        cpd = TabularCPD().fit(X, y)
        probs = np.asarray(cpd.predict_proba(X[:5]).probs)
        expected = np.log(probs[np.arange(5), (y["y"][:5] == "1").astype(int)])
        scores = cpd.log_likelihood(X[:5].set_axis(["r"] * 5), y[:5].set_axis(["r"] * 5))
        assert (scores.index.tolist(), scores.columns.tolist()) == (["r"] * 5, ["y"])
        np.testing.assert_allclose(scores["y"], expected)
        assert cpd.log_likelihood(X[:1], pd.Series(["9"], index=X.index[:1], name="y"))["y"].iloc[0] == -np.inf
        root = TabularCPD().fit(None, y)
        expected = np.log(np.where(y["y"][:2] == "0", 0.56, 0.44))
        np.testing.assert_allclose(root.log_likelihood(None, y[:2])["y"], expected)

    def test_sample(self, discrete_data):
        X, y = discrete_data

        root = TabularCPD().fit(None, y)
        samples = root.sample(n_samples=5, random_state=0)
        assert samples.shape == (5, 1) and list(samples.columns) == ["y"]
        pd.testing.assert_frame_equal(samples, root.sample(n_samples=5, random_state=0))

        cpd = TabularCPD().fit(X, y)
        X_test = X[:3].set_axis(["a", "b", "c"])
        assert cpd.sample(X_test, random_state=0).index.tolist() == ["a", "b", "c"]
        assert cpd.sample(X_test, n_samples=2, random_state=0).shape == (6, 1)

    def test_from_values(self, discrete_data):
        X, y = discrete_data
        fitted = TabularCPD().fit(X, y)

        # The arguments follow pgmpy.factors.discrete.TabularCPD. A table given with the parents in another order,
        # cyclic ones too, is reordered to the sorted order.
        swapped = fitted.CPT_.reshape(2, 3, 2).transpose(0, 2, 1).reshape(2, 6)
        cpd = TabularCPD.from_values("y", 2, swapped, ["x2", "x1"], [2, 3], fitted.state_names_)
        assert cpd.is_fitted and cpd == fitted and hash(cpd) == hash(fitted)
        np.testing.assert_allclose(np.asarray(cpd.predict_proba(X).probs), np.asarray(fitted.predict_proba(X).probs))
        values = np.random.default_rng(0).dirichlet([1, 1], size=24).T
        cyclic = TabularCPD.from_values("y", 2, values, ["c", "a", "b"], [2, 3, 4])
        assert cyclic.evidence_ == ["a", "b", "c"]
        np.testing.assert_allclose(cyclic.CPT_, values.reshape(2, 2, 3, 4).transpose(0, 2, 3, 1).reshape(2, 24))

        # The table is copied, so reusing the caller's array doesn't change the CPD.
        values = np.array([[0.56], [0.44]])
        root = TabularCPD.from_values("y", 2, values, state_names={"y": ["0", "1"]})
        values[:] = 0.5
        assert root == TabularCPD().fit(None, y)

        # CPDs compare by distribution, whatever the order of their states.
        flipped = fitted.CPT_.reshape(2, 3, 2)[::-1, ::-1, :].reshape(2, 6)
        state_names = {"y": ["1", "0"], "x1": ["2", "1", "0"], "x2": ["0", "1"]}
        reordered = TabularCPD.from_values("y", 2, flipped, ["x1", "x2"], [3, 2], state_names)
        assert reordered == fitted and hash(reordered) == hash(fitted)
        assert TabularCPD.from_values("y", 2, fitted.CPT_, ["x1", "x2"], [3, 2], state_names) != fitted

        # As in pgmpy.factors.discrete.TabularCPD, variables without state names get the states 0, 1, ...
        legacy = LegacyTabularCPD("y", 2, [[0.2, 0.7], [0.8, 0.3]], ["x"], [2])
        cpd = TabularCPD.from_values("y", 2, [[0.2, 0.7], [0.8, 0.3]], ["x"], [2])
        assert cpd.state_names_ == legacy.state_names == {"y": [0, 1], "x": [0, 1]}
        partial = TabularCPD.from_values("y", 2, [[0.2, 0.7], [0.8, 0.3]], ["x"], [2], {"y": ["a", "b"]})
        assert partial.state_names_ == {"y": ["a", "b"], "x": [0, 1]}

        with pytest.raises(ValueError, match="shape"):
            TabularCPD.from_values("y", 2, [[0.5, 0.5]])
        with pytest.raises(ValueError, match="sum to 1"):
            TabularCPD.from_values("y", 2, [[0.5], [0.4]])
        with pytest.raises(ValueError, match="states"):
            TabularCPD.from_values("y", 3, [[0.5], [0.5]], state_names={"y": ["0", "1"]})
        with pytest.raises(ValueError, match="evidence_card"):
            TabularCPD.from_values("y", 2, [[0.5], [0.5]], ["x"])
        with pytest.raises(ValueError, match="Repeated state names"):
            TabularCPD.from_values("y", 2, [[0.5], [0.5]], state_names={"y": ["a", "a"]})
        for evidence in (["y"], ["x", "x"]):
            with pytest.raises(ValueError, match="different names"):
                TabularCPD.from_values("y", 2, np.full((2, 2 ** len(evidence)), 0.5), evidence, [2] * len(evidence))
        with pytest.raises(TypeError, match="list or tuple"):
            TabularCPD.from_values("y", 2, [[0.5], [0.5]], {"x"}, [1])
