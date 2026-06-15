import numpy as np
import pandas as pd

from pgmpy.ci_tests import GeneralizedCov
from pgmpy.tests.test_ci_tests import _multivariate_fixtures

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
    test = GeneralizedCov(data=df_square, n_permutations=10, random_state=0)
    test.run_test("X", "Y", [])

    res_x, _ = test.get_residuals("X", [])
    res_y, _ = test.get_residuals("Y", [])
    expected = _manual_generalized_cov_statistic(res_x, res_y)

    assert test.statistic_ == expected


def test_generalized_cov_statistic_matches_rectangular_generalization(pillai_data):
    df_rectangular = pillai_data["indep"][2]
    test = GeneralizedCov(data=df_rectangular, n_permutations=10, random_state=0)
    test.run_test("X", "Y", ["Z1", "Z2", "Z3"])

    res_x, _ = test.get_residuals("X", ["Z1", "Z2", "Z3"])
    res_y, _ = test.get_residuals("Y", ["Z1", "Z2", "Z3"])
    expected = _manual_generalized_cov_statistic(res_x, res_y)

    assert test.statistic_ == expected


def test_effect_size(pillai_data):
    df = pillai_data["indep"][0]
    test = GeneralizedCov(data=df, n_permutations=10, random_state=0)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.effect_size_ is None


def test_generalized_cov_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = GeneralizedCov(data=df, n_permutations=500, random_state=0)
        test("X", "Y", [])
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"]:
        test = GeneralizedCov(data=df, n_permutations=500, random_state=0)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = GeneralizedCov(data=df, n_permutations=500, random_state=0)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.p_value_ <= 0.05
