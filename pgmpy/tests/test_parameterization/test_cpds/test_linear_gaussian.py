import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from skbase.utils.dependencies import _check_soft_dependencies

import pgmpy.parameterization
from pgmpy.example_models import load_model
from pgmpy.parameter_estimator import LinearGaussianMLE
from pgmpy.parameterization import LinearGaussianCPD, LinearGaussianOLS

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


def ols_cpd(std_estimator="unbiased"):
    return LinearGaussianCPD(estimator=LinearGaussianOLS(std_estimator=std_estimator))


def assert_same_fit(cpd, expected):
    assert (cpd.variable_, cpd.evidence_) == (expected.variable_, expected.evidence_)
    np.testing.assert_allclose([*cpd.beta_, cpd.std_], [*expected.beta_, expected.std_], rtol=1e-9, atol=1e-12)


class TestLinearGaussianCPD:
    def test_tags(self):
        assert LinearGaussianCPD.get_class_tag("object_type") == "parameterization"
        assert LinearGaussianCPD.get_class_tag("variable_type") == ["continuous"]
        assert LinearGaussianCPD.get_class_tag("parent_data_types") == ["continuous"]
        assert LinearGaussianCPD.get_class_tag("capability:exact_inference") is True
        assert LinearGaussianCPD.get_class_tag("capability:factor") is False
        for assumption in ("linearity", "additive_noise", "gaussian_noise"):
            assert LinearGaussianCPD.get_class_tag(f"assumption:{assumption}") is True
        assert LinearGaussianCPD.get_class_tag("supports_weighted_data") is True
        assert LinearGaussianCPD.get_class_tag("python_dependencies") == "skpro"
        assert pgmpy.parameterization.LinearGaussianCPD is LinearGaussianCPD

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
        assert ols_cpd("mle").fit(X, y).std_ == pytest.approx(np.sqrt(rss / len(y)))

        # A root variable has its mean and standard deviation.
        root = LinearGaussianCPD().fit(None, y)
        np.testing.assert_allclose(root.beta_, [y.mean()])
        assert root.std_ == pytest.approx(y.std(ddof=1))
        assert ols_cpd("mle").fit(None, y).std_ == pytest.approx(y.std(ddof=0))

        # Sample weights count as frequencies: integer weights give the fit of each row repeated that many times. The
        # "mle" std also doesn't change when all weights are scaled.
        weights = np.random.default_rng(0).integers(0, 4, size=len(y))
        repeated = X.index.repeat(weights)
        for std_estimator in ("unbiased", "mle"):
            for X_fit, X_repeated in ((X, X.loc[repeated]), (None, None)):
                weighted = ols_cpd(std_estimator).fit(X_fit, y, sample_weight=weights)
                assert_same_fit(weighted, ols_cpd(std_estimator).fit(X_repeated, y.loc[repeated]))
        assert_same_fit(ols_cpd("mle").fit(X, y, weights / 1000), ols_cpd("mle").fit(X, y, weights))

        # Rows with zero weight don't count, even when their squares would overflow.
        overflowing, first_dropped = y.where(y.index != 0, 1e155), np.r_[0.0, np.ones(len(y) - 1)]
        weighted = LinearGaussianCPD().fit(X, overflowing, sample_weight=first_dropped)
        assert_same_fit(weighted, LinearGaussianCPD().fit(X.iloc[1:], y.iloc[1:]))

        # The unbiased std counts independent coefficients, so a repeated or a constant parent doesn't change it.
        X_extra = X.assign(C=X["A"], D=5.0)
        extra = LinearGaussianCPD().fit(X_extra, y)
        assert extra.std_ == pytest.approx(cpd.std_)
        np.testing.assert_allclose(extra.predict_proba(X_extra).mean(), cpd.predict_proba(X).mean())

        constant = pd.Series(5.0, index=y.index, name="y")
        four = pd.Series([1.0, 2.0, 4.0, 3.0], name="y")
        rejected = [
            (LinearGaussianCPD(), X[:3], y[:3], None, "total weight"),  # as many rows as coefficients
            (LinearGaussianCPD(), X, y, weights / 1000, "total weight"),  # frequencies below the coefficients
            (LinearGaussianCPD(), None, four, [0.2, 0.4, 0.3, 0.1], "total weight"),  # a total of 1 up to rounding
            (LinearGaussianCPD(), None, constant, None, "constant"),  # a constant root has std 0
            (LinearGaussianCPD(), X, constant, None, "constant"),  # so does a constant child
            (LinearGaussianCPD(), None, pd.Series(0.1, index=range(12), name="y"), None, "constant"),  # mean rounds
            (LinearGaussianCPD(), None, pd.Series([1e200, -1e200, 3e200], name="y"), None, "finite"),  # std overflows
            (LinearGaussianCPD(), None, y.where(y.index != 3, np.inf), None, "finite"),  # inf in y gives a NaN std
            (LinearGaussianCPD(), X, y.astype(str), None, "numeric"),
            (LinearGaussianCPD(), X.assign(A=X["A"].astype(str)), y, None, "numeric"),
            (LinearGaussianCPD(), X.assign(A=X["A"].round().astype("category")), y, None, "numeric"),
            (LinearGaussianCPD(), X.assign(A=pd.Timestamp("2020-01-01")), y, None, "datatype"),
        ]
        for cpd_bad, X_bad, y_bad, weights_bad, match in rejected:
            with pytest.raises(ValueError, match=match), np.errstate(over="ignore"):
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

        # Parents of any numeric dtype, such as booleans and nullable integers, give float means.
        X_mixed = X.assign(A=X["A"].round().astype("Int64"), B=X["B"] > 0)
        mixed = LinearGaussianCPD().fit(X_mixed, y).predict_proba(X_mixed)
        assert mixed.mean()["y"].dtype == np.float64 and np.isfinite(mixed.log_pdf(y.to_frame())).all().all()

        # A root gives its own distribution for every row, or as a single distribution without X.
        root = LinearGaussianCPD().fit(None, y)
        rows = root.predict_proba(pd.DataFrame(index=[7, 8]))
        assert rows.index.tolist() == [7, 8]
        np.testing.assert_allclose(rows.mean()["y"], [y.mean()] * 2)
        dist = root.predict_proba()
        assert dist.shape == () and dist.mean() == pytest.approx(y.mean())
        assert dist.var() == pytest.approx(y.var(ddof=1))

        with pytest.raises(ValueError):
            cpd.predict_proba(X[["A"]])

    def test_predict(self, data):
        # The mean of each row, with X's index, whatever the order of X's columns; a root predicts its mean.
        X, y = data
        cpd = LinearGaussianCPD().fit(X, y)
        rows = X.iloc[:4].set_axis(["a", "b", "c", "d"])
        expected = pd.DataFrame({"y": cpd.beta_[0] + rows[["A", "B"]].to_numpy() @ cpd.beta_[1:]}, index=rows.index)
        pd.testing.assert_frame_equal(cpd.predict(rows[["B", "A"]]), expected)
        root = LinearGaussianCPD().fit(None, y)
        pd.testing.assert_frame_equal(root.predict(pd.DataFrame(index=[7])), pd.DataFrame({"y": [y.mean()]}, index=[7]))

    def test_log_likelihood(self, data):
        # The Normal log-density of y around each row's mean; a root is scored with X=None.
        X, y = data
        cpd = LinearGaussianCPD().fit(X, y)
        means = cpd.beta_[0] + X[["A", "B"]].to_numpy() @ cpd.beta_[1:]
        np.testing.assert_allclose(cpd.log_likelihood(X, y)["y"], norm.logpdf(y, means, cpd.std_))
        root = LinearGaussianCPD().fit(None, y)
        np.testing.assert_allclose(root.log_likelihood(None, y)["y"], norm.logpdf(y, y.mean(), root.std_))

    def test_sample(self, data):
        X, y = data
        cpd = LinearGaussianCPD().fit(X, y)
        rows = X.iloc[:3].set_axis(["a", "b", "c"])

        # Samples keep X's index, and n_samples adds a first index level numbering the draws.
        samples = cpd.sample(rows, random_state=0)
        assert (samples.index.tolist(), samples.columns.tolist()) == (["a", "b", "c"], ["y"])
        drawn = cpd.sample(rows, n_samples=2, random_state=0)
        assert drawn.index.tolist() == [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")]
        assert cpd.sample(rows, n_samples=0).shape == (0, 1)

        # An int seed gives the same values on every call, a Generator advances, and None draws fresh values.
        pd.testing.assert_frame_equal(samples, cpd.sample(rows, random_state=0))
        rng = np.random.default_rng(0)
        assert not cpd.sample(rows, random_state=rng).equals(cpd.sample(rows, random_state=rng))
        assert not cpd.sample(rows).equals(cpd.sample(rows))

        # Rows are matched by position, so repeated labels and a MultiIndex, like a parent's own draws, keep them apart.
        repeated = cpd.sample(X.iloc[[0, 0, 1]], n_samples=2, random_state=0)
        assert repeated.index.tolist() == [(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 1)]
        by_position = cpd.sample(X.iloc[[0, 0, 1]].reset_index(drop=True), n_samples=2, random_state=0)
        np.testing.assert_array_equal(repeated["y"], by_position["y"])
        multi = X.iloc[:3].set_axis(pd.MultiIndex.from_tuples([(0, "a"), (0, "b"), (1, "a")]))
        np.testing.assert_array_equal(cpd.sample(multi, random_state=0)["y"], samples["y"])

        # The values follow the predicted distribution: given X, here A = 1 and B = 0 in every row, or for a root
        # without X, its own distribution.
        n = 20000
        root = LinearGaussianCPD().fit(None, y)
        X_one = pd.DataFrame({"A": np.ones(n), "B": np.zeros(n)})
        for values, mean, var in (
            (cpd.sample(X_one, random_state=0)["y"], cpd.beta_[0] + cpd.beta_[1], cpd.std_**2),
            (root.sample(n_samples=n, random_state=0)["y"], root.beta_[0], root.std_**2),
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

        # Parents are sorted, also from a cyclic order, and numbers come before strings.
        for evidence, sorted_evidence, sorted_beta in (
            (["c", "a", "b"], ["a", "b", "c"], [0, 2, 3, 1]),
            (["b", 0], [0, "b"], [0, 2, 1]),
            ([10, 9], [9, 10], [0, 2, 1]),
        ):
            reordered = LinearGaussianCPD.from_values("y", range(len(evidence) + 1), 1.0, evidence)
            assert (reordered.evidence_, reordered.beta_.tolist()) == (sorted_evidence, sorted_beta)

        # A root has its own distribution without data.
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
            ("y", [0.0], 10**400, None),  # too large for a float
            ("y", np.array([1 + 1j]), 1.0, None),  # complex beta
            ("y", [0.0, 1.0], 1.0, ["y"]),  # the variable as its own parent
            ("y", [0.0, 1.0, 2.0], 1.0, ["A", "A"]),  # a repeated parent
        ]
        for variable, beta_bad, std_bad, evidence in rejected:
            with pytest.raises(ValueError):
                LinearGaussianCPD.from_values(variable, beta_bad, std_bad, evidence)
        with pytest.raises(TypeError):
            LinearGaussianCPD.from_values(["y"], [0.0], 1.0)
        with pytest.raises(TypeError, match="list or tuple"):
            LinearGaussianCPD.from_values("y", [0.0, 1.0, 2.0], 1.0, {"a", "b"})

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
                cpd = ols_cpd(std_estimator).fit(
                    data[legacy.evidence] if legacy.evidence else None, data[legacy.variable]
                )
                reference = LinearGaussianCPD.from_values(legacy.variable, legacy.beta, legacy.std, legacy.evidence)
                assert_same_fit(cpd, reference)
