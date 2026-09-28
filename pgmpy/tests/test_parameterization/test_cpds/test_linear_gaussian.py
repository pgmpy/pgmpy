import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.example_models import load_model
from pgmpy.factors.continuous import LinearGaussianCPD as LegacyLinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import LinearGaussianMLE
from pgmpy.parameterization.cpds import LinearGaussianCPD

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


@pytest.fixture
def data():
    """y = 1 + 2 A - 3 B + N(0, 0.5^2), with the parents' columns in unsorted order."""
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame({"B": rng.normal(size=500), "A": rng.normal(1, 2, size=500)})
    y = pd.Series(1 + 2 * X["A"] - 3 * X["B"] + rng.normal(scale=0.5, size=500), name="y")
    return X, y


def assert_same_fit(cpd, expected):
    assert (cpd.variable_, cpd.evidence_) == (expected.variable_, expected.evidence_)
    np.testing.assert_allclose([*cpd.beta_, cpd.std_], [*expected.beta_, expected.std_], rtol=1e-9, atol=1e-12)


class TestLinearGaussianCPD:
    def test_tags(self):
        assert LinearGaussianCPD.get_class_tag("object_type") == "parameterization"
        assert LinearGaussianCPD.get_class_tag("variable_type") == "continuous"
        assert LinearGaussianCPD.get_class_tag("parent_type") == "continuous"
        assert LinearGaussianCPD.get_class_tag("supports_weighted_data") is True
        assert LinearGaussianCPD.get_class_tag("python_dependencies") == "skpro"

    def test_fit(self, data):
        X, y = data
        design = np.column_stack([np.ones(len(y)), X["A"], X["B"]])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        rss = np.sum((y - design @ beta) ** 2)

        # Parents are sorted by name, and beta_ follows them. The default "unbiased" std divides the sum of squared
        # residuals by the number of rows minus the number of coefficients, "mle" by the number of rows.
        cpd = LinearGaussianCPD().fit(X, y)
        assert (cpd.variable_, cpd.evidence_) == ("y", ["A", "B"])
        np.testing.assert_allclose(cpd.beta_, beta, rtol=1e-10)
        assert cpd.std_ == pytest.approx(np.sqrt(rss / (len(y) - 3)))
        assert LinearGaussianCPD(std_estimator="mle").fit(X, y).std_ == pytest.approx(np.sqrt(rss / len(y)))

        # A root variable has its mean and standard deviation.
        root = LinearGaussianCPD().fit(None, y)
        np.testing.assert_allclose(root.beta_, [y.mean()])
        assert root.std_ == pytest.approx(y.std(ddof=1))
        assert LinearGaussianCPD(std_estimator="mle").fit(None, y).std_ == pytest.approx(y.std(ddof=0))

        # Sample weights count as frequencies: integer weights give the fit, and marginal, of each row repeated that
        # many times. The "mle" std also doesn't change when all weights are scaled.
        weights = np.random.default_rng(0).integers(0, 4, size=len(y))
        repeated = X.index.repeat(weights)
        for std_estimator in ("unbiased", "mle"):
            for X_fit, X_repeated in ((X, X.loc[repeated]), (None, None)):
                weighted = LinearGaussianCPD(std_estimator).fit(X_fit, y, sample_weight=weights)
                expected = LinearGaussianCPD(std_estimator).fit(X_repeated, y.loc[repeated])
                assert_same_fit(weighted, expected)
                marginal, expected_marginal = weighted.predict_proba(), expected.predict_proba()
                assert marginal.mean() == pytest.approx(expected_marginal.mean())
                assert marginal.var() == pytest.approx(expected_marginal.var())
        assert_same_fit(LinearGaussianCPD("mle").fit(X, y, weights / 1000), LinearGaussianCPD("mle").fit(X, y, weights))

        # The unbiased std counts independent coefficients, so a repeated or a constant parent doesn't change it.
        X_extra = X.assign(C=X["A"], D=5.0)
        extra = LinearGaussianCPD().fit(X_extra, y)
        assert extra.std_ == pytest.approx(cpd.std_)
        np.testing.assert_allclose(extra.predict_proba(X_extra).mean(), cpd.predict_proba(X).mean())

        constant = pd.Series(5.0, index=y.index, name="y")
        rejected = [
            (LinearGaussianCPD(std_estimator="unbias"), X, y, None, "std_estimator"),
            (LinearGaussianCPD(), X[:3], y[:3], None, "total weight"),  # as many rows as coefficients
            (LinearGaussianCPD(), X, y, weights / 1000, "total weight"),  # frequencies below the coefficients
            (LinearGaussianCPD(), None, constant, None, "positive"),  # a constant root has std 0
            (LinearGaussianCPD(), X, constant, None, "positive"),  # so does a constant child
            (LinearGaussianCPD(), None, y.where(y.index != 3, np.inf), None, "infinite"),
            (LinearGaussianCPD(), X, y.astype(str), None, "numeric"),
            (LinearGaussianCPD(), X.assign(A=X["A"].astype(str)), y, None, "numeric"),
            (LinearGaussianCPD(), X.assign(A=X["A"].round().astype("category")), y, None, "numeric"),
            (LinearGaussianCPD(), X.assign(A=pd.Timestamp("2020-01-01")), y, None, "numeric"),
        ]
        for cpd_bad, X_bad, y_bad, weights_bad, match in rejected:
            with pytest.raises(ValueError, match=match):
                cpd_bad.fit(X_bad, y_bad, sample_weight=weights_bad)

    def test_predict_proba(self, data):
        X, y = data
        cpd = LinearGaussianCPD().fit(X, y)

        # One Normal per row of X, whose columns can come in any order, with the fitted mean and standard deviation.
        rows = X.iloc[:4].set_axis(["a", "b", "c", "d"])
        dist = cpd.predict_proba(rows)
        assert (dist.index.tolist(), dist.columns.tolist()) == (["a", "b", "c", "d"], ["y"])
        np.testing.assert_allclose(dist.mean()["y"], cpd.beta_[0] + rows[["A", "B"]].to_numpy() @ cpd.beta_[1:])
        np.testing.assert_allclose(dist.var()["y"], cpd.std_**2)

        # A root gives its own distribution for every row, or as a single distribution without X.
        root = LinearGaussianCPD().fit(None, y)
        rows = root.predict_proba(pd.DataFrame(index=[7, 8]))
        assert rows.index.tolist() == [7, 8]
        np.testing.assert_allclose(rows.mean()["y"], [y.mean()] * 2)
        marginal = root.predict_proba()
        assert marginal.shape == () and marginal.mean() == pytest.approx(y.mean())
        assert marginal.var() == pytest.approx(y.var(ddof=1))

        for X_bad in (X.assign(A=np.inf), X.assign(A=X["A"].astype(str)), X[["A"]]):
            with pytest.raises(ValueError):
                cpd.predict_proba(X_bad)

    def test_marginal(self):
        # Without X, a child gives the marginal distribution that the model implies for the parent data seen in fit. For
        # a network with every edge, fitted with "mle", that is the network's own joint Gaussian. With "unbiased", the
        # parents' covariance divides by the number of rows minus one, so it matches for parents that are roots.
        rng = np.random.default_rng(0)
        A = rng.exponential(2, size=300)
        B = 1 - A + rng.normal(size=300)
        data = pd.DataFrame({"A": A, "B": B, "C": 2 + 0.5 * A - B + rng.normal(size=300)})
        parents = {"A": [], "B": ["A"], "C": ["A", "B"]}
        for std_estimator, checked in (("mle", ["A", "B", "C"]), ("unbiased", ["A", "B"])):
            cpds = {
                node: LinearGaussianCPD(std_estimator).fit(data[evidence] if evidence else None, data[node])
                for node, evidence in parents.items()
            }
            network = LinearGaussianBayesianNetwork([("A", "B"), ("A", "C"), ("B", "C")])
            network.add_cpds(
                *(LegacyLinearGaussianCPD(node, cpd.beta_, cpd.std_, cpd.evidence_) for node, cpd in cpds.items())
            )
            mean, cov = network.to_joint_gaussian()
            for position, node in enumerate(["A", "B", "C"]):
                if node in checked:
                    marginal = cpds[node].predict_proba()
                    assert marginal.mean() == pytest.approx(mean[position])
                    assert marginal.var() == pytest.approx(cov[position, position])

    def test_sample(self, data):
        X, y = data
        cpd = LinearGaussianCPD().fit(X, y)
        rows = X.iloc[:3].set_axis(["a", "b", "c"])

        # Samples keep X's index, and n_samples adds a first index level numbering the draws.
        samples = cpd.sample(rows, random_state=0)
        assert (samples.index.tolist(), samples.columns.tolist()) == (["a", "b", "c"], ["y"])
        drawn = cpd.sample(rows, n_samples=2, random_state=0)
        assert drawn.index.tolist() == [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")]
        assert cpd.sample(rows, n_samples=0).shape == cpd.sample(n_samples=0).shape == (0, 1)

        # An int seed gives the same values on every call, a Generator advances, and None draws fresh values.
        pd.testing.assert_frame_equal(samples, cpd.sample(rows, random_state=0))
        rng = np.random.default_rng(0)
        assert not cpd.sample(rows, random_state=rng).equals(cpd.sample(rows, random_state=rng))
        assert not cpd.sample(rows).equals(cpd.sample(rows))

        # The values follow the predicted distribution: given X, here A = 1 and B = 0 in every row, or without X, the
        # marginal distribution.
        n = 20000
        marginal = cpd.predict_proba()
        X_one = pd.DataFrame({"A": np.ones(n), "B": np.zeros(n)})
        for values, mean, var in (
            (cpd.sample(X_one, random_state=0)["y"], cpd.beta_[0] + cpd.beta_[1], cpd.std_**2),
            (cpd.sample(n_samples=n, random_state=0)["y"], marginal.mean(), marginal.var()),
        ):
            assert values.mean() == pytest.approx(mean, abs=4 * np.sqrt(var / n))
            assert values.std() == pytest.approx(np.sqrt(var), rel=0.03)

    def test_from_values(self, data):
        X, y = data
        fitted = LinearGaussianCPD().fit(X, y)

        # The arguments follow pgmpy.factors.continuous.LinearGaussianCPD. Parents given in another order are sorted,
        # with beta reordered to match.
        cpd = LinearGaussianCPD.from_values("y", fitted.beta_[[0, 2, 1]], fitted.std_, ["B", "A"])
        assert cpd.is_fitted and (cpd.evidence_, cpd.std_) == (["A", "B"], fitted.std_)
        np.testing.assert_array_equal(cpd.beta_, fitted.beta_)
        assert cpd == fitted and hash(cpd) == hash(fitted)
        np.testing.assert_array_equal(cpd.predict_proba(X).mean(), fitted.predict_proba(X).mean())

        # Without data, only a root has a marginal distribution.
        with pytest.raises(ValueError, match="from_values"):
            cpd.predict_proba()
        root = LinearGaussianCPD.from_values("A", [2.0], 0.5)
        assert (root.predict_proba().mean(), root.predict_proba().var()) == (2.0, 0.25)

        # beta is copied, so changing the caller's array, or the CPD it came from, leaves the new CPD as it was.
        beta = fitted.beta_.copy()
        copied = LinearGaussianCPD.from_values("y", beta, fitted.std_, fitted.evidence_)
        beta[:] = 0
        assert copied == fitted and not np.shares_memory(copied.beta_, beta)

        rejected = [
            ("y", [1.0, 2.0], 1.0, None),  # beta longer than 1 + parents
            ("y", [1.0], 1.0, ["A"]),  # beta shorter
            ("y", [np.nan], 1.0, None),
            ("y", [np.inf], 1.0, None),
            ("y", [0.0], 0.0, None),
            ("y", [0.0], -0.0, None),
            ("y", [0.0], -1.0, None),
            ("y", [0.0], np.inf, None),
            ("y", [0.0], np.nan, None),
            ("y", [0.0], np.array([1.0]), None),
            ("y", [0.0, 1.0], 1.0, ["y"]),  # the variable as its own parent
            ("y", [0.0, 1.0, 2.0], 1.0, ["A", "A"]),  # a repeated parent
        ]
        for variable, beta_bad, std_bad, evidence in rejected:
            with pytest.raises(ValueError):
                LinearGaussianCPD.from_values(variable, beta_bad, std_bad, evidence)
        with pytest.raises(TypeError):
            LinearGaussianCPD.from_values(["y"], [0.0], 1.0)

    def test_equality(self, data):
        X, y = data
        cpd = LinearGaussianCPD().fit(X, y)

        # Parents are matched by name, and floats within numpy.allclose's tolerance count as equal, so refits on
        # reordered rows or columns compare equal.
        for other in (
            LinearGaussianCPD().fit(X[["A", "B"]], y),
            LinearGaussianCPD().fit(X.iloc[::-1], y.iloc[::-1]),
            LinearGaussianCPD.from_values("y", cpd.beta_, cpd.std_ * (1 + 1e-12), cpd.evidence_),
        ):
            assert cpd == other and hash(cpd) == hash(other)
        assert LinearGaussianCPD.from_values("y", cpd.beta_, cpd.std_ * 1.1, cpd.evidence_) != cpd
        assert LinearGaussianCPD().fit(X[:100], y[:100]) != cpd

    def test_matches_linear_gaussian_mle(self):
        model = load_model("bnlearn/ecoli70")
        data = model.simulate(2000, seed=0)
        for std_estimator in ("unbiased", "mle"):
            expected = LinearGaussianMLE(std_estimator=std_estimator).fit(model, data).parameters_
            assert len(expected) == 46
            for legacy in expected:
                cpd = LinearGaussianCPD(std_estimator).fit(
                    data[legacy.evidence] if legacy.evidence else None, data[legacy.variable]
                )
                reference = LinearGaussianCPD.from_values(legacy.variable, legacy.beta, legacy.std, legacy.evidence)
                assert_same_fit(cpd, reference)
