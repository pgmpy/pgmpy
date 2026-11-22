import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from pgmpy.ci_tests import GeneralizedCov, PillaiTrace
from pgmpy.tests.test_ci_tests import _multivariate_fixtures
from pgmpy.utils import CovarianceRegression

pillai_data = _multivariate_fixtures.pillai_data


def _manual_generalized_cov_statistic(res_x, res_y):
    if isinstance(res_x, pd.Series):
        res_x = res_x.to_frame()
    if isinstance(res_y, pd.Series):
        res_y = res_y.to_frame()

    x = res_x.to_numpy(dtype=float)
    y = res_y.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross_cov = (x.T @ y) / (x.shape[0] - 1)
    return float(np.prod(np.linalg.svd(cross_cov, compute_uv=False)))


def test_generalized_cov_statistic_matches_cross_covariance_determinant_for_square_case(pillai_data):
    df_square = pillai_data["indep"][3]
    test = GeneralizedCov(data=df_square, n_permutations=10, rng=0)
    test.run_test("X", "Y", [])

    res_x, _ = test.get_residuals("X", [])
    res_y, _ = test.get_residuals("Y", [])
    expected = _manual_generalized_cov_statistic(res_x, res_y)

    assert test.statistic_ == expected


def test_generalized_cov_statistic_matches_rectangular_generalization(pillai_data):
    df_rectangular = pillai_data["indep"][2]
    test = GeneralizedCov(data=df_rectangular, n_permutations=10, rng=0)
    test.run_test("X", "Y", ["Z1", "Z2", "Z3"])

    res_x, _ = test.get_residuals("X", ["Z1", "Z2", "Z3"])
    res_y, _ = test.get_residuals("Y", ["Z1", "Z2", "Z3"])
    expected = _manual_generalized_cov_statistic(res_x, res_y)

    assert test.statistic_ == expected


@pytest.mark.parametrize(("variant", "Z"), [(0, ["Z1", "Z2", "Z3"]), (2, ["Z1", "Z2", "Z3"]), (3, [])])
def test_effect_size_is_rms_canonical_correlation(pillai_data, variant, Z):
    df = pillai_data["indep"][variant]
    generalized_cov = GeneralizedCov(data=df, n_permutations=10, rng=0)
    generalized_cov("X", "Y", Z)
    pillai = PillaiTrace(data=df)
    pillai("X", "Y", Z)

    assert generalized_cov.effect_size_**2 == pytest.approx(pillai.effect_size_, abs=1e-4)


def test_generalized_cov_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = GeneralizedCov(data=df, n_permutations=500, rng=0)
        test("X", "Y", [])
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"]:
        test = GeneralizedCov(data=df, n_permutations=500, rng=0)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = GeneralizedCov(data=df, n_permutations=500, rng=0)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.p_value_ <= 0.05


def test_default_estimator_types():
    rng = np.random.default_rng(0)
    n = 1000
    Z = rng.normal(size=n)
    X_cont = Z * 0.5 + rng.normal(size=n)
    Y_cont = Z * 0.5 + rng.normal(size=n)
    X_cat = pd.cut(X_cont, bins=4, labels=["a", "b", "c", "d"])

    # Continuous variables: estimator should be CovarianceRegression.
    test = GeneralizedCov(data=pd.DataFrame({"X": X_cont, "Y": Y_cont, "Z": Z}), n_permutations=10, rng=0)
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, CovarianceRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)

    # Categorical X: estimator_x_ should be LogisticRegression, estimator_y_ CovarianceRegression.
    test = GeneralizedCov(data=pd.DataFrame({"X": X_cat, "Y": Y_cont, "Z": Z}), n_permutations=10, rng=0)
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, LogisticRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)


def test_linearity_tag():
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n), "Z": rng.normal(size=n)})

    assert GeneralizedCov.get_class_tag("assumption:linearity") is False
    assert GeneralizedCov(data=data).get_tag("assumption:linearity") is True
    assert (
        GeneralizedCov(data=data, estimator=RandomForestRegressor(random_state=0)).get_tag("assumption:linearity")
        is False
    )
