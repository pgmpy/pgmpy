from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from pgmpy.structure_score import AICGauss, BICGauss, GeneralizedStructureScore, LogLikelihoodGauss


@pytest.fixture
def linear_gaussian_data():
    """Ground truth: y = 2*x + noise, noise ~ N(0, 1)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    y = 2 * x + rng.standard_normal(500)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def spline_lr():
    """CAM-style estimator: spline features + linear regression."""
    return Pipeline(
        [
            ("spline", SplineTransformer(degree=3, n_knots=5)),
            ("lr", LinearRegression()),
        ]
    )


class TestInit:
    def test_defaults(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data)
        assert isinstance(s.estimator, LinearRegression)
        assert s.penalty == "bic"
        assert s.noise_dist is stats.norm
        assert s.n_params is None


class TestPenaltyValue:
    @pytest.mark.parametrize(
        ("penalty", "expected"),
        [(None, 0), ("aic", 10), ("bic", 5 * np.log(100)), pytest.param(lambda k, n: 3 * k, 15, id="callable")],
    )
    def test_penalty_value(self, linear_gaussian_data, penalty, expected):
        s = GeneralizedStructureScore(linear_gaussian_data, penalty=penalty)
        assert s.penalty is penalty
        assert s._penalty_value(5, 100) == pytest.approx(expected)

    def test_invalid_penalty(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, penalty="invalid")
        with pytest.raises(ValueError, match="Unknown penalty"):
            s.local_score("y", ("x",))


class TestNParams:
    def test_sklearn_linear_regression(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        X = linear_gaussian_data[["x"]].to_numpy()
        est = LinearRegression().fit(X, linear_gaussian_data["y"].to_numpy())
        # 1 coef + 1 intercept = 2
        assert s._n_params(est, X) == 2

    def test_custom_n_params(self, linear_gaussian_data):
        n_params = lambda est, X: 99
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), n_params=n_params)
        X = linear_gaussian_data[["x"]].to_numpy()
        est = LinearRegression().fit(X, linear_gaussian_data["y"].to_numpy())
        assert s._n_params(est, X) == 99
        assert s.get_params()["n_params"] is n_params
        assert "GeneralizedStructureScore" in repr(s)

        original_score = s.local_score("y", ("x",))
        s.set_params(n_params=lambda est, X: 100)
        assert s.local_score("y", ("x",)) == pytest.approx(original_score - np.log(len(X)) / 2)

    @pytest.mark.parametrize(
        ("attributes", "expected"),
        [pytest.param({"df_model": 3}, 4, id="statsmodels"), pytest.param({"statistics_": {"edof": 7}}, 7, id="pygam")],
    )
    def test_parameter_count_conventions(self, linear_gaussian_data, attributes, expected):
        s = GeneralizedStructureScore(linear_gaussian_data)
        assert s._n_params(SimpleNamespace(**attributes), np.zeros((10, 2))) == expected

    def test_unknown_estimator_raises(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data)
        with pytest.raises(AttributeError, match="Cannot determine parameter count"):
            s._n_params(SimpleNamespace(), np.zeros((10, 2)))


class TestLocalScore:
    def test_callable_penalty(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, penalty=lambda k, n: 10 * k * np.log(n))
        log_likelihood = LogLikelihoodGauss(linear_gaussian_data).local_score("y", ("x",))
        assert s.local_score("y", ("x",)) == pytest.approx(log_likelihood - 15 * np.log(len(linear_gaussian_data)))

    @pytest.mark.parametrize("penalty", [None, "aic"])
    def test_spline_estimator(self, linear_gaussian_data, spline_lr, penalty):
        s = GeneralizedStructureScore(linear_gaussian_data, spline_lr, penalty=penalty)
        assert np.isfinite(s.local_score("y", ("x",)))

    @pytest.mark.parametrize(
        ("penalty", "reference"), [(None, LogLikelihoodGauss), ("aic", AICGauss), ("bic", BICGauss)]
    )
    @pytest.mark.parametrize("parents", [(), ("x",), ("x", "z")])
    def test_matches_gaussian_scores(self, linear_gaussian_data, penalty, reference, parents):
        data = linear_gaussian_data.assign(z=np.random.default_rng(1).standard_normal(len(linear_gaussian_data)))
        s = GeneralizedStructureScore(data, penalty=penalty)
        result = s.local_score("y", parents)

        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result == pytest.approx(reference(data).local_score("y", parents))
        assert s.local_score("y", parents) == result

    @pytest.mark.parametrize("noise_dist", [stats.norm, stats.laplace])
    @pytest.mark.parametrize(("fit_intercept", "use_pipeline"), [(True, False), (False, False), (True, True)])
    def test_counts_location_once(self, linear_gaussian_data, noise_dist, fit_intercept, use_pipeline):
        X = linear_gaussian_data[["x"]].to_numpy()
        y = linear_gaussian_data["y"].to_numpy()
        estimator = LinearRegression(fit_intercept=fit_intercept)
        if use_pipeline:
            estimator = Pipeline([("scale", StandardScaler()), ("regression", estimator)])
        fitted = estimator.fit(X, y)
        residuals = y - fitted.predict(X)
        log_likelihood = noise_dist.logpdf(residuals, *noise_dist.fit(residuals)).sum()
        expected = log_likelihood - 1.5 * np.log(len(y))
        score = GeneralizedStructureScore(linear_gaussian_data, estimator, noise_dist=noise_dist)

        assert score.noise_dist is noise_dist
        assert score.local_score("y", ("x",)) == pytest.approx(expected)

    def test_preserves_configured_pygam_terms(self):
        pygam = pytest.importorskip("pygam")
        x = np.linspace(-3, 3, 80)
        y = np.sin(x) + np.random.default_rng(5).normal(scale=0.1, size=len(x))
        data = pd.DataFrame({"x": x, "y": y})
        estimator = pygam.LinearGAM(pygam.s(0, n_splines=8))
        reference = pygam.LinearGAM(pygam.s(0, n_splines=8))
        reference.fit(x[:, None], y)
        residuals = y - reference.predict(x[:, None])
        expected = stats.norm.logpdf(residuals, *stats.norm.fit(residuals)).sum()
        score = GeneralizedStructureScore(data, estimator, penalty=None)

        assert score.local_score("y", ("x",)) == pytest.approx(expected)
        assert not hasattr(estimator, "coef_")


class TestCausalDirection:
    def test_cam_prefers_true_direction(self, spline_lr):
        """Spline scoring prefers x -> y for y = x**2 + Gaussian noise."""
        rng = np.random.default_rng(0)
        n = 500
        x = rng.standard_normal(n)
        y = x**2 + 0.3 * rng.standard_normal(n)
        data = pd.DataFrame({"x": x, "y": y})

        s = GeneralizedStructureScore(data, spline_lr, penalty=None)

        # True DAG: x -> y
        score_true = s.local_score("y", ("x",)) + s.local_score("x", ())
        # Reversed DAG: y -> x
        score_reverse = s.local_score("x", ("y",)) + s.local_score("y", ())

        assert score_true > score_reverse

    def test_laplace_prefers_true_direction(self):
        """Laplace scoring prefers x -> y for y = 2*x + Laplace noise."""
        rng = np.random.default_rng(1)
        n = 300
        x = rng.standard_normal(n)
        y = 2 * x + rng.laplace(scale=0.5, size=n)
        data = pd.DataFrame({"x": x, "y": y})

        s = GeneralizedStructureScore(data, LinearRegression(), noise_dist=stats.laplace, penalty="bic")
        score_true = s.local_score("y", ("x",)) + s.local_score("x", ())
        score_reverse = s.local_score("x", ("y",)) + s.local_score("y", ())

        assert score_true > score_reverse
