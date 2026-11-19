import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from pgmpy.utils import CovarianceRegression


def test_methods():
    rng = np.random.default_rng(seed=0)
    n = 500
    given = rng.normal(size=(n, 3)) * [1e-6, 1.0, 1e5] + [1e3, -2.0, 7e5]
    targets = given @ [[3e5, -1.0], [2.0, 0.5], [1e-5, 4e-6]] + rng.normal(size=(n, 2))
    data = pd.DataFrame(np.column_stack([targets, given]), columns=[0, ("t", 1), "g1", 2, ("g", 3)])
    X, y = ["g1", 2, ("g", 3)], [0, ("t", 1)]
    regression = CovarianceRegression().fit(X=data)

    centered_given = given - given.mean(axis=0)
    centered_targets = targets - targets.mean(axis=0)
    coefficients = np.linalg.lstsq(centered_given, centered_targets, rcond=None)[0].T
    residuals = centered_targets - centered_given @ coefficients.T

    assert regression.feature_names_in_ == list(data.columns)
    assert regression.n_samples_ == n
    np.testing.assert_allclose(regression.location_, data.mean().to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(regression.coefficients(X=X, y=y), coefficients, rtol=1e-8)
    np.testing.assert_allclose(
        regression.residual_covariance(X=X, y=y), np.cov(residuals, rowvar=False, ddof=0), rtol=1e-8
    )
    np.testing.assert_allclose(
        regression.partial_correlation(X=X, y=y), np.corrcoef(residuals, rowvar=False), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(regression.residual_covariance(X=[], y=y), regression.covariance_[:2, :2])

    estimator = LedoitWolf()
    shrunk = CovarianceRegression(covariance_estimator=estimator).fit(X=data)
    assert not hasattr(estimator, "covariance_")
    np.testing.assert_allclose(shrunk.covariance_, LedoitWolf().fit(data.to_numpy()).covariance_, rtol=1e-12)


def test_singular():
    rng = np.random.default_rng(seed=1)
    x = rng.normal(size=300)
    values = np.column_stack([3.0 * x + rng.normal(size=300), x, x, np.full(300, 5.0)])
    regression = CovarianceRegression().fit(X=values)
    centered = values - values.mean(axis=0)
    coefficients = np.linalg.lstsq(centered[:, 1:], centered[:, :1], rcond=None)[0].T
    residuals = centered[:, :1] - centered[:, 1:] @ coefficients.T

    assert regression.feature_names_in_ == [0, 1, 2, 3]
    np.testing.assert_allclose(regression.coefficients(X=[1, 2, 3], y=[0]), coefficients, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        regression.residual_covariance(X=[1, 2, 3], y=[0]), np.cov(residuals, rowvar=False, ddof=0), rtol=1e-10
    )
