"""
Tests for GeneralizedStructureScore.

File location in repo: pgmpy/tests/test_structure_score/test_generalized_structure_score.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from pgmpy.structure_score import GeneralizedStructureScore


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
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        assert s.penalty == "bic"
        assert s.noise_dist is stats.norm
        assert s.n_params_fn is None

    def test_custom_noise_dist(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), noise_dist=stats.laplace)
        assert s.noise_dist is stats.laplace

    def test_none_penalty(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty=None)
        assert s.penalty is None

    def test_callable_penalty_stored(self, linear_gaussian_data):
        fn = lambda k, n: k * 2.0
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty=fn)
        assert s.penalty is fn


class TestPenaltyValue:
    def setup_method(self):
        data = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        self.s = GeneralizedStructureScore(data, LinearRegression())

    def test_none_returns_zero(self):
        self.s.penalty = None
        assert self.s._penalty_value(5, 100) == 0.0

    def test_aic_returns_k(self):
        self.s.penalty = "aic"
        assert self.s._penalty_value(5, 100) == 5.0

    def test_bic_returns_k_log_n(self):
        self.s.penalty = "bic"
        assert self.s._penalty_value(5, 100) == pytest.approx(5 * np.log(100))

    def test_callable_is_called(self):
        self.s.penalty = lambda k, n: k * 3.0
        assert self.s._penalty_value(4, 100) == pytest.approx(12.0)

    def test_invalid_raises(self):
        self.s.penalty = "invalid"
        with pytest.raises(ValueError, match="Unknown penalty"):
            self.s._penalty_value(5, 100)


class TestNParams:
    def test_sklearn_linear_regression(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        X = linear_gaussian_data[["x"]].to_numpy()
        est = LinearRegression().fit(X, linear_gaussian_data["y"].to_numpy())
        # 1 coef + 1 intercept = 2
        assert s._n_params(est, X) == 2

    def test_custom_n_params_fn(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), n_params=lambda est, X: 99)
        X = linear_gaussian_data[["x"]].to_numpy()
        est = LinearRegression().fit(X, linear_gaussian_data["y"].to_numpy())
        assert s._n_params(est, X) == 99

    def test_statsmodels_convention(self, linear_gaussian_data):
        """Estimator with df_model attribute (statsmodels-like)."""

        class FakeStatsmodels:
            df_model = 3

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        s = GeneralizedStructureScore(linear_gaussian_data, FakeStatsmodels())
        est = FakeStatsmodels()
        X = np.zeros((10, 2))
        # df_model=3 + 1 intercept = 4
        assert s._n_params(est, X) == 4

    def test_pygam_convention(self, linear_gaussian_data):
        """Estimator with statistics_['edof'] attribute (pygam-like)."""

        class FakePygam:
            statistics_ = {"edof": 7}

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        s = GeneralizedStructureScore(linear_gaussian_data, FakePygam())
        est = FakePygam()
        X = np.zeros((10, 2))
        assert s._n_params(est, X) == 7

    def test_unknown_estimator_raises(self, linear_gaussian_data):
        class WeirdEst:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        s = GeneralizedStructureScore(linear_gaussian_data, WeirdEst())
        est = WeirdEst()
        X = np.zeros((10, 2))
        with pytest.raises(AttributeError, match="Cannot determine parameter count"):
            s._n_params(est, X)


class TestLocalScore:
    def test_returns_float(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        result = s.local_score("y", ("x",))
        assert isinstance(result, float)

    def test_finite_with_parents(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        assert np.isfinite(s.local_score("y", ("x",)))

    def test_finite_no_parents(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        assert np.isfinite(s.local_score("y", ()))

    def test_bic_lower_than_aic(self, linear_gaussian_data):
        """BIC penalises more than AIC for n > e^2 ≈ 7.4."""
        s_aic = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty="aic")
        s_bic = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty="bic")
        assert s_bic.local_score("y", ("x",)) < s_aic.local_score("y", ("x",))

    def test_laplace_noise(self, linear_gaussian_data):
        """LiNGAM-flavoured: Laplace noise should run without error."""
        s = GeneralizedStructureScore(
            linear_gaussian_data,
            LinearRegression(),
            noise_dist=stats.laplace,
            penalty="bic",
        )
        assert np.isfinite(s.local_score("y", ("x",)))

    def test_callable_penalty(self, linear_gaussian_data):
        """Custom penalty callable should run and return finite score."""
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty=lambda k, n: k * 3.0)
        assert np.isfinite(s.local_score("y", ("x",)))

    def test_invalid_penalty_raises_on_call(self, linear_gaussian_data):
        """Invalid penalty string should raise ValueError when scoring."""
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty="invalid")
        with pytest.raises(ValueError, match="Unknown penalty"):
            s.local_score("y", ("x",))

    def test_spline_estimator(self, linear_gaussian_data, spline_lr):
        """CAM-style spline pipeline should work end to end."""
        s = GeneralizedStructureScore(linear_gaussian_data, spline_lr, penalty=None)
        assert np.isfinite(s.local_score("y", ("x",)))

    def test_aic_with_spline(self, linear_gaussian_data, spline_lr):
        """TOPIC-style: spline + AIC should return finite score."""
        s = GeneralizedStructureScore(linear_gaussian_data, spline_lr, penalty="aic")
        assert np.isfinite(s.local_score("y", ("x",)))

    def test_parents_as_tuple(self, linear_gaussian_data):
        """BaseStructureScore passes parents as tuple — must work."""
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        # tuple input (as called by the base class via lru_cache)
        result = s._local_score("y", ("x",))
        assert np.isfinite(result)

    def test_caching_consistent(self, linear_gaussian_data):
        """Calling local_score twice should return the same value (lru_cache)."""
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        r1 = s.local_score("y", ("x",))
        r2 = s.local_score("y", ("x",))
        assert r1 == r2

    def test_exact_value_aic(self, linear_gaussian_data):
        """Verify exact AIC score value with manual computation."""
        data = linear_gaussian_data
        y = data["y"].to_numpy()
        X = data[["x"]].to_numpy()

        est = LinearRegression().fit(X, y)
        residuals = y - est.predict(X)
        params = stats.norm.fit(residuals)
        log_L = float(stats.norm.logpdf(residuals, *params).sum())
        k = 4  # 1 coef + 1 intercept + 2 norm params
        expected = log_L - k / 2

        s = GeneralizedStructureScore(data, LinearRegression(), penalty="aic")
        assert s.local_score("y", ("x",)) == pytest.approx(expected, rel=1e-5)

    def test_exact_value_no_penalty(self, linear_gaussian_data):
        """Verify exact score with no penalty."""
        data = linear_gaussian_data
        y = data["y"].to_numpy()
        X = data[["x"]].to_numpy()

        est = LinearRegression().fit(X, y)
        residuals = y - est.predict(X)
        params = stats.norm.fit(residuals)
        log_L = float(stats.norm.logpdf(residuals, *params).sum())

        s = GeneralizedStructureScore(data, LinearRegression(), penalty=None)
        assert s.local_score("y", ("x",)) == pytest.approx(log_L, rel=1e-5)

    def test_exact_value_no_parents(self, linear_gaussian_data):
        """Verify exact marginal score with no parents."""
        data = linear_gaussian_data
        y = data["y"].to_numpy()
        n = len(y)

        params = stats.norm.fit(y)
        log_L = float(stats.norm.logpdf(y, *params).sum())
        k = 2  # 2 norm params
        expected = log_L - (k * np.log(n)) / 2

        s = GeneralizedStructureScore(data, LinearRegression(), penalty="bic")
        assert s.local_score("y", ()) == pytest.approx(expected, rel=1e-5)


class TestCAMRecovery:
    """
    CAM-style scoring should prefer the true causal direction over the reverse
    on nonlinear synthetic data.

    Data generating process: y = x^2 + noise (x -> y).
    Total DAG score = sum of local scores for all nodes.
    """

    def test_cam_prefers_true_direction(self):
        rng = np.random.default_rng(0)
        n = 500
        x = rng.standard_normal(n)
        y = x**2 + 0.3 * rng.standard_normal(n)
        data = pd.DataFrame({"x": x, "y": y})

        spline_lr = Pipeline(
            [
                ("spline", SplineTransformer(degree=3, n_knots=5)),
                ("lr", LinearRegression()),
            ]
        )
        s = GeneralizedStructureScore(data, spline_lr, penalty=None)

        # True DAG: x -> y
        score_true = s.local_score("y", ("x",)) + s.local_score("x", ())
        # Reversed DAG: y -> x
        score_reverse = s.local_score("x", ("y",)) + s.local_score("y", ())

        assert score_true > score_reverse, (
            f"CAM should prefer x->y (true) over y->x (reverse). Got true={score_true:.2f}, reverse={score_reverse:.2f}"
        )

    def test_lingam_laplace_runs_on_nongaussian(self):
        """LiNGAM-flavoured scoring on Laplace-distributed noise."""
        rng = np.random.default_rng(1)
        n = 300
        x = rng.standard_normal(n)
        # Laplace noise
        y = 2 * x + rng.laplace(scale=0.5, size=n)
        data = pd.DataFrame({"x": x, "y": y})

        s = GeneralizedStructureScore(data, LinearRegression(), noise_dist=stats.laplace, penalty="bic")
        score_true = s.local_score("y", ("x",)) + s.local_score("x", ())
        score_reverse = s.local_score("x", ("y",)) + s.local_score("y", ())

        # True direction should score higher
        assert score_true > score_reverse


class TestEdgeCases:
    def test_single_parent(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        assert np.isfinite(s.local_score("y", ("x",)))

    def test_empty_parents_tuple(self, linear_gaussian_data):
        s = GeneralizedStructureScore(linear_gaussian_data, LinearRegression())
        assert np.isfinite(s.local_score("y", ()))

    def test_score_decreases_with_stronger_bic_penalty(self, linear_gaussian_data):
        """Stronger penalty → lower score."""
        s_bic = GeneralizedStructureScore(linear_gaussian_data, LinearRegression(), penalty="bic")
        s_strong = GeneralizedStructureScore(
            linear_gaussian_data,
            LinearRegression(),
            penalty=lambda k, n: k * np.log(n) * 10,
        )
        assert s_strong.local_score("y", ("x",)) < s_bic.local_score("y", ("x",))
