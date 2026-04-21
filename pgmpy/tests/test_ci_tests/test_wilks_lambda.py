import numpy as np

from pgmpy.ci_tests import WilksLambda

from pgmpy.tests.test_ci_tests._multivariate_fixtures import pillai_data, skip_gh_actions


@skip_gh_actions
def test_wilks_no_cond(pillai_data):
    expected_stats = [0.8428, 0.8428, 0.8477, 0.8561, 0.8477]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", [])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_wilks_indep(pillai_data):
    expected_stats = [0.9984, 0.9993, 0.9980, 0.9863, 0.9980]
    expected_pvalues = [0.2125, 0.4154, 0.5741, 0.1338, 0.5741]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_wilks_dependent(pillai_data):
    expected_stats = [0.8300, 0.7841, 0.8283, 0.7914, 0.8283]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


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
