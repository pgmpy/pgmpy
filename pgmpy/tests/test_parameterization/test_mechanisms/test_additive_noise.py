import numpy as np
import pandas as pd
import pytest
from scipy.stats import laplace
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from pgmpy.parameterization import (
    AdditiveNoiseMechanism,
    DeterministicAdapter,
    DistributionAdapter,
    LinearGaussianCPD,
    SklearnAdapter,
    SkproAdapter,
    TabularCPD,
)

GLMRegressor = _safe_import("skpro.regression.linear.GLMRegressor")
Laplace = _safe_import("skpro.distributions.Laplace")
Poisson = _safe_import("skpro.distributions.Poisson")
ResidualDouble = _safe_import("skpro.regression.residual.ResidualDouble")

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


def f(X):
    return np.sin(X["a"]) + 2 * X["b"]


@pytest.fixture(scope="module")
def data():
    """y = sin(a) + 2 b + Laplace(0, 0.5), whose noise has standard deviation 0.707, on rows labelled with strings."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"a": rng.uniform(-3, 3, size=2000), "b": rng.normal(size=2000)}, index=[f"r{i}" for i in range(2000)]
    )
    return X, (f(X) + rng.laplace(scale=0.5, size=2000)).rename("y")


class TestAdditiveNoiseMechanism:
    def test_slots_and_tags(self):
        # f comes from any parameterization of a continuous target. The noise comes from one that gives a
        # distribution: by default a LinearGaussianCPD, so Gaussian noise.
        default = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression()))
        assert default.get_tag("variable_type") == ["continuous"] and default.get_tag("assumption:additive_noise")
        assert default.get_tag("supports_weighted_data") and default.get_tag("assumption:gaussian_noise")
        assert default.get_tag("parent_data_types") == ["discrete", "continuous", "mixed"]
        assert not default.get_tag("capability:exact_inference")

        # The tags follow the slots: a linear f with Gaussian noise is a linear Gaussian model, and a skpro noise takes
        # no weights.
        linear = AdditiveNoiseMechanism(LinearGaussianCPD())
        assert linear.get_tag("capability:exact_inference") and linear.get_tag("parent_data_types") == ["continuous"]
        skpro_noise = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression()), SkproAdapter(GLMRegressor()))
        assert not skpro_noise.get_tag("supports_weighted_data")
        assert not skpro_noise.get_tag("assumption:gaussian_noise")

        for function, noise, match in (
            (TabularCPD(), None, "function"),
            (SklearnAdapter(LogisticRegression()), None, "function"),
            (LinearRegression(), None, "function"),
            (LinearGaussianCPD(), SklearnAdapter(LinearRegression()), "distribution"),
        ):
            with pytest.raises(TypeError, match=match):
                AdditiveNoiseMechanism(function, noise)

    def test_known_mechanism(self, data):
        # A known function with known Laplace noise: each row gets the Laplace located at f(x).
        X, _ = data
        mechanism = AdditiveNoiseMechanism.from_values("y", f, Laplace(mu=0.0, scale=0.5), evidence=["b", "a"])
        assert mechanism.is_fitted and mechanism.evidence_ == ["a", "b"]
        rows = X.iloc[:4]
        values = f(rows).to_numpy() + np.array([-1.0, -0.2, 0.3, 1.5])
        dist = mechanism.predict_proba(rows)
        np.testing.assert_allclose(dist.mean()["y"], f(rows))
        np.testing.assert_allclose(dist.cdf(values.reshape(-1, 1))["y"], laplace.cdf(values, f(rows), 0.5))
        observed = pd.Series(values, index=rows.index, name="y")
        np.testing.assert_allclose(mechanism.log_likelihood(rows, observed)["y"], laplace.logpdf(values, f(rows), 0.5))
        np.testing.assert_allclose(mechanism.predict(rows)["y"], f(rows))
        draws = mechanism.sample(rows.iloc[[0]], n_samples=20000, random_state=0)["y"] - f(rows.iloc[[0]]).iloc[0]
        assert draws.mean() == pytest.approx(0.0, abs=0.02) and draws.var() == pytest.approx(0.5, rel=0.05)

    def test_fit(self, data):
        X, y = data
        # f is the function's prediction, and the default noise is the Normal with the residuals' mean and standard
        # deviation.
        mechanism = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression())).fit(X, y)
        residuals = y - mechanism.function_.predict(X)["y"]
        rows = X.iloc[:3]
        dist = mechanism.predict_proba(rows)
        np.testing.assert_allclose(dist.mean()["y"], mechanism.function_.predict(rows)["y"] + residuals.mean())
        np.testing.assert_allclose(dist.var()["y"], residuals.var(ddof=1))
        np.testing.assert_allclose(mechanism.predict(rows)["y"], dist.mean()["y"])

        # A Laplace noise, from a skpro regressor fitted as a root on the residuals, recovers the noise's scale.
        # skpro 2.14 renamed ResidualDouble's distr_type to dist.
        linear_y = (1 + 2 * X["a"] - X["b"] + y - f(X)).rename("y")
        param = "dist" if "dist" in ResidualDouble.get_param_names() else "distr_type"
        noise = SkproAdapter(ResidualDouble(LinearRegression(), **{param: "Laplace"}))
        fitted = AdditiveNoiseMechanism(LinearGaussianCPD(), noise).fit(X, linear_y).noise_.predict_proba()
        assert np.sqrt(fitted.var() / 2) == pytest.approx(0.5, rel=0.05)

    def test_cv(self, data):
        # A flexible function memorizes its training rows, so its in-sample residuals understate the noise, whose
        # standard deviation is 0.707; out-of-fold residuals don't.
        X, y = data
        forest = SklearnAdapter(RandomForestRegressor(n_estimators=30, random_state=0))
        in_sample = AdditiveNoiseMechanism(forest).fit(X, y).noise_.std_
        out_of_fold = AdditiveNoiseMechanism(forest, cv=5).fit(X, y).noise_.std_
        assert in_sample < 0.5 < out_of_fold

        # A function that reproduces its training data leaves no residuals to fit the noise on.
        with pytest.raises(ValueError, match="cv"):
            AdditiveNoiseMechanism(SklearnAdapter(KNeighborsRegressor(n_neighbors=1))).fit(X, y)

    def test_structural_methods(self, data):
        X, y = data
        mechanism = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression())).fit(X, y)
        rows, observed = X.iloc[:5], y.iloc[:5]

        # A unit's noise is its residual, and the forward pass on it gives the observation back.
        noise = mechanism.abduct(rows, observed)
        np.testing.assert_allclose(noise["y"], observed - mechanism.function_.predict(rows)["y"])
        pd.testing.assert_frame_equal(mechanism.predict(rows, noise=noise), observed.to_frame())

        # With shared noise, two interventions differ only through f.
        u = mechanism.sample_noise(len(rows), random_state=0)
        effect = mechanism.predict(rows.assign(a=1.0), noise=u) - mechanism.predict(rows.assign(a=0.0), noise=u)
        f_effect = mechanism.function_.predict(rows.assign(a=1.0)) - mechanism.function_.predict(rows.assign(a=0.0))
        np.testing.assert_allclose(effect, f_effect)

        # A root is its noise around a constant f.
        root = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression())).fit(None, y)
        assert root.predict_proba().mean() == pytest.approx(y.mean())
        assert root.sample(n_samples=5, random_state=0).shape == (5, 1)

    def test_weights_and_noise(self, data):
        X, y = data
        weights = np.random.default_rng(1).uniform(0.5, 2, size=len(y))
        AdditiveNoiseMechanism(SklearnAdapter(LinearRegression())).fit(X, y, sample_weight=weights)
        with pytest.raises(ValueError, match="sample_weight"):
            AdditiveNoiseMechanism(SklearnAdapter(LinearRegression()), SkproAdapter(GLMRegressor())).fit(
                X, y, sample_weight=weights
            )

        # A discrete noise distribution, such as a Poisson, can't be added to a continuous target.
        with pytest.raises(ValueError, match="discrete"):
            AdditiveNoiseMechanism(SklearnAdapter(LinearRegression()), DistributionAdapter(Poisson(mu=2.0))).fit(X, y)
        with pytest.raises(ValueError, match="discrete"):
            AdditiveNoiseMechanism.from_values("y", f, Poisson(mu=2.0), evidence=["a", "b"])

        # A known function with a learned noise: fitting the function warns that it ignores the data.
        with pytest.warns(UserWarning, match="ignores the data"):
            known = AdditiveNoiseMechanism(DeterministicAdapter(f)).fit(X, y)
        assert known.noise_.std_ == pytest.approx(np.std(y - f(X), ddof=1))
