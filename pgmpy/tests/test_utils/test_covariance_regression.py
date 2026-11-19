import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from pgmpy.utils import CovarianceRegression


def test_matches_least_squares_on_badly_scaled_data():
    rng = np.random.default_rng(seed=0)
    n = 500
    given = rng.normal(size=(n, 3)) * [1e-6, 1.0, 1e5] + [1e3, -2.0, 7e5]
    targets = given @ [[3e5, -1.0], [2.0, 0.5], [1e-5, 4e-6]] + rng.normal(size=(n, 2))
    data = pd.DataFrame(np.column_stack([targets, given]), columns=[0, ("t", 1), "g1", 2, ("g", 3)])
    regression = CovarianceRegression().fit(X=data)

    centered_given = given - given.mean(axis=0)
    centered_targets = targets - targets.mean(axis=0)
    expected = np.linalg.lstsq(centered_given, centered_targets, rcond=None)[0]
    residuals = centered_targets - centered_given @ expected

    assert regression.feature_names_in_ == [0, ("t", 1), "g1", 2, ("g", 3)]
    assert regression.n_samples_ == n
    np.testing.assert_allclose(regression.location_, data.mean().to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(regression.coefficients(X=["g1", 2, ("g", 3)], y=[0, ("t", 1)]), expected.T, rtol=1e-8)
    np.testing.assert_allclose(
        regression.residual_covariance(X=["g1", 2, ("g", 3)], y=[0, ("t", 1)]),
        np.cov(residuals, rowvar=False, ddof=0),
        rtol=1e-8,
    )
    np.testing.assert_array_equal(regression.residual_covariance(X=[], y=[0, ("t", 1)]), regression.covariance_[:2, :2])


def test_singular_regressors_give_minimum_norm_fit():
    rng = np.random.default_rng(seed=1)
    x = rng.normal(size=300)
    values = np.column_stack([3.0 * x + rng.normal(size=300), x, x, np.full(300, 5.0)])
    regression = CovarianceRegression().fit(X=values)
    centered = values - values.mean(axis=0)

    assert regression.feature_names_in_ == [0, 1, 2, 3]
    np.testing.assert_allclose(
        regression.coefficients(X=[1, 2, 3], y=[0]),
        np.linalg.lstsq(centered[:, 1:], centered[:, :1], rcond=None)[0].T,
        rtol=1e-10,
        atol=1e-12,
    )


def test_covariance_estimator_is_cloned_and_used():
    rng = np.random.default_rng(seed=2)
    data = pd.DataFrame(rng.normal(size=(40, 4)), columns=["A", "B", "C", "D"])
    estimator = LedoitWolf()
    regression = CovarianceRegression(covariance_estimator=estimator).fit(X=data)
    expected = LedoitWolf().fit(data.to_numpy())

    assert not hasattr(estimator, "covariance_")
    assert regression.covariance_estimator_ is not estimator
    np.testing.assert_allclose(regression.covariance_, expected.covariance_, rtol=1e-12)
    np.testing.assert_allclose(regression.location_, expected.location_, rtol=1e-12)

    cov = expected.covariance_
    coefficients = np.linalg.solve(cov[1:, 1:], cov[1:, :1])
    np.testing.assert_allclose(regression.coefficients(X=["B", "C", "D"], y=["A"]), coefficients.T, rtol=1e-10)
    np.testing.assert_allclose(
        regression.residual_covariance(X=["B", "C", "D"], y=["A"]),
        cov[:1, :1] - cov[:1, 1:] @ coefficients,
        rtol=1e-10,
    )
