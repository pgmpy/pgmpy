from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, poisson
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from pgmpy.parameterization.adapters import SkproAdapter

BootstrapRegressor = _safe_import("skpro.regression.bootstrap.BootstrapRegressor")
GLMRegressor = _safe_import("skpro.regression.linear.GLMRegressor")
HistBinnedProbaRegressor = _safe_import("skpro.regression.binned.HistBinnedProbaRegressor")
ResidualDouble = _safe_import("skpro.regression.residual.ResidualDouble")

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


@pytest.fixture(scope="module")
def data():
    """Y = 1 + 2 A - 3 B + N(0, 0.5^2), on rows labelled with strings."""
    rng = np.random.default_rng(42)
    X, X_test = (
        pd.DataFrame(rng.normal(size=(n, 2)), columns=["A", "B"], index=[f"{name}{i}" for i in range(n)])
        for name, n in (("train", 500), ("test", 100))
    )
    y = pd.Series(1 + 2 * X["A"] - 3 * X["B"] + rng.normal(scale=0.5, size=len(X)), name="Y")
    return X, y, X_test, 1 + 2 * X_test["A"] - 3 * X_test["B"]


class TestSkproAdapter:
    def test_tags(self):
        assert SkproAdapter.get_class_tag("variable_type") == ["continuous"]
        assert SkproAdapter.get_class_tag("supports_weighted_data") is False
        assert SkproAdapter.get_class_tag("python_dependencies") == "skpro"
        for estimator in ("not an estimator", LogisticRegression()):
            with pytest.raises(TypeError, match="skpro"):
                SkproAdapter(estimator)

    def test_fit_and_predict(self, data):
        X, y, X_test, mean = data
        estimator = GLMRegressor(add_constant=True)
        adapter = SkproAdapter(estimator).fit(X, y)

        # A clone is fitted, and each row of X gets its predicted distribution, labelled with the target. This GLM
        # recovers the mean and the noise.
        assert adapter.estimator_ is not estimator and not estimator.is_fitted
        dist = adapter.predict_proba(X_test)
        assert type(dist).__name__ == "Normal"
        assert (dist.index.tolist(), dist.columns.tolist()) == (X_test.index.tolist(), ["Y"])
        pd.testing.assert_frame_equal(dist.mean(), adapter.estimator_.predict_proba(X_test).mean())
        pd.testing.assert_frame_equal(adapter.predict(X_test), dist.mean())
        expected = adapter.estimator_.predict_proba(X).log_pdf(y.to_frame())["Y"].to_numpy()
        np.testing.assert_allclose(adapter.log_likelihood(X, y)["Y"], expected)
        assert np.sqrt(np.mean((dist.mean()["Y"] - mean) ** 2)) < 0.1
        np.testing.assert_allclose(np.sqrt(dist.var()["Y"]), 0.5, rtol=0.1)

        # Predictions must have a row per row of X.
        binned = SkproAdapter(HistBinnedProbaRegressor.create_test_instance()).fit(X, y)
        with pytest.raises(ValueError, match="HistBinnedProbaRegressor"):
            binned.predict_proba(X_test.iloc[[3]])

        # Samples follow the seed and X's rows, also when X has none.
        samples = adapter.sample(X_test, n_samples=3, random_state=0)
        assert samples.shape == (300, 1) and samples.columns.tolist() == ["Y"]
        pd.testing.assert_frame_equal(samples, adapter.sample(X_test, n_samples=3, random_state=0))
        assert not samples.equals(adapter.sample(X_test, n_samples=3, random_state=1))
        assert adapter.sample(X_test.iloc[:0]).shape == (0, 1)

        # skpro regressors take no sample weights.
        with pytest.raises(ValueError, match="sample_weight"):
            SkproAdapter(GLMRegressor()).fit(X, y, sample_weight=np.ones(len(y)))

        # A root is the estimator fitted on a constant column, the same model without parents, so it can be scored. This
        # GLM gives the Normal with y's mean and standard deviation, and a Poisson GLM the Poisson with the mean count.
        root = SkproAdapter(GLMRegressor()).fit(None, y)
        dist = root.predict_proba()
        assert type(dist).__name__ == "Normal" and dist.ndim == 0
        assert (dist.mean(), dist.var()) == pytest.approx((y.mean(), y.var(ddof=1)))
        np.testing.assert_allclose(root.predict_proba(pd.DataFrame(index=["p", "q"])).mean()["Y"], [y.mean()] * 2)
        np.testing.assert_allclose(root.log_likelihood(None, y)["Y"], norm.logpdf(y, y.mean(), y.std(ddof=1)))
        assert root.sample(n_samples=10, random_state=0).shape == (10, 1)
        counts = pd.Series(np.random.default_rng(0).poisson(3.0, size=200).astype(float), name="C")
        counted = SkproAdapter(GLMRegressor(family="Poisson")).fit(None, counts)
        np.testing.assert_allclose(counted.log_likelihood(None, counts)["C"], poisson.logpmf(counts, counts.mean()))

        # A fitted estimator compares by identity, so a fitted adapter only equals itself.
        assert adapter == adapter and len({adapter, adapter}) == 1
        assert adapter != SkproAdapter(GLMRegressor(add_constant=True)).fit(X, y)

    def test_inputs(self, data):
        X, y, _, _ = data

        # skpro rejects object and string columns, so they go in as categories, with the categories seen in fit.
        group = pd.Series(np.where(X["A"] > 0, "yes", "no"), index=X.index, name="G")
        target = (2.0 * (group == "yes") + y / 10).rename("Y")
        model = SkproAdapter(ResidualDouble(make_pipeline(OneHotEncoder(), LinearRegression())))
        model.fit(group.to_frame(), target)
        means = model.predict_proba(pd.DataFrame({"G": ["yes", "no"]}, index=["y", "n"])).mean()["Y"]
        np.testing.assert_allclose(means, target.groupby(group).mean()[["yes", "no"]])
        with pytest.raises(ValueError, match="'maybe'"):
            model.predict_proba(pd.DataFrame({"G": ["no", "maybe"]}))

        # Numeric and boolean parents of any dtype go in as floats, as GLMRegressor needs, and so does the target.
        plain = SkproAdapter(GLMRegressor(add_constant=True)).fit(X, y)
        dtypes = ["Float64"] + (["double[pyarrow]"] if find_spec("pyarrow") else [])
        for X_typed in [X.astype(dtype) for dtype in dtypes] + [X.assign(B=X["B"] > 0).astype({"B": "boolean"})]:
            typed = SkproAdapter(GLMRegressor(add_constant=True)).fit(X_typed, y.astype("Float64"))
            expected = SkproAdapter(GLMRegressor(add_constant=True)).fit(X_typed.astype(float), y)
            expected_means = expected.predict_proba(X_typed[:3].astype(float)).mean()
            np.testing.assert_allclose(typed.predict_proba(X_typed[:3]).mean(), expected_means)
        root = SkproAdapter(GLMRegressor()).fit(None, y.astype("Float64"))
        assert root.predict_proba().mean() == pytest.approx(y.mean())

        # skpro matches rows by label, so the adapter fits by position, and predict, sample and log_likelihood predict
        # by position: data with repeated labels, or a MultiIndex like a parent's draws, works. predict_proba keeps X's
        # labels, so it raises for repeated ones.
        multi = pd.MultiIndex.from_arrays([np.arange(len(y)) % 5, np.arange(len(y)) // 5])
        root = SkproAdapter(GLMRegressor()).fit(None, y.set_axis(multi))
        rows, repeated_rows = pd.DataFrame(index=multi[:3]), pd.DataFrame(index=multi[[0, 0, 1]])
        dist = root.predict_proba(rows)
        assert dist.index.equals(rows.index)
        np.testing.assert_allclose(dist.mean()["Y"], [y.mean()] * 3)
        assert root.sample(repeated_rows, random_state=0).index.equals(repeated_rows.index)
        with pytest.raises(ValueError, match="unique labels"):
            root.predict_proba(repeated_rows)
        repeated = SkproAdapter(GLMRegressor(add_constant=True)).fit(
            X.set_axis(["r"] * len(X)), y.set_axis(["r"] * len(y))
        )
        X_repeated = X[:3].set_axis(["a", "a", "b"])
        with pytest.raises(ValueError, match="unique labels"):
            repeated.predict_proba(X_repeated)
        np.testing.assert_allclose(repeated.predict(X_repeated)["Y"], plain.predict(X[:3])["Y"])
        assert repeated.sample(X_repeated, random_state=0).index.tolist() == ["a", "a", "b"]
        bootstrap = SkproAdapter(BootstrapRegressor(LinearRegression(), n_bootstrap_samples=20, random_state=0)).fit(
            X, y
        )
        np.testing.assert_allclose(bootstrap.predict(X_repeated)["Y"], bootstrap.predict(X[:3])["Y"])

        # Names that aren't strings, which skpro regressors can't take, go in as strings. Samples keep the real name.
        X_named = X.set_axis([0, ("B", 1)], axis=1)
        named = SkproAdapter(GLMRegressor(add_constant=True)).fit(X_named, y.rename(("Y", 0)))
        np.testing.assert_allclose(named.predict_proba(X_named[:3]).mean(), plain.predict_proba(X[:3]).mean())
        assert named.sample(X_named[:3], random_state=0).columns.tolist() == [("Y", 0)]
