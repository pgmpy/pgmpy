import numpy as np

from pgmpy.ci_tests import WilksLambda
from pgmpy.tests.test_ci_tests import _multivariate_fixtures

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_wilks_no_cond(pillai_data):
    expected_stats = [0.8384, 0.8384, 0.8771, 0.8996, 0.8771]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", [])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


@skip_gh_actions
def test_wilks_indep(pillai_data):
    expected_stats = [0.9974, 0.9996, 0.9997, 0.9975, 0.9997]
    expected_pvalues = [0.1072, 0.5069, 0.8774, 0.6434, 0.8774]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


@skip_gh_actions
def test_wilks_dependent(pillai_data):
    expected_stats = [0.8300, 0.7841, 0.8664, 0.8464, 0.8664]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


def test_wilks_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", [])
        assert test.statistic_ < 1.0
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ > 0.9
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ < 0.95
        assert test.p_value_ <= 0.05
