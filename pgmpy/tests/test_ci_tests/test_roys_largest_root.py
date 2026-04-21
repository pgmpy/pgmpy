import numpy as np

from pgmpy.ci_tests import RoysLargestRoot

from pgmpy.tests.test_ci_tests._multivariate_fixtures import pillai_data, skip_gh_actions


@skip_gh_actions
def test_roys_no_cond(pillai_data):
    expected_stats = [0.1572, 0.1572, 0.1523, 0.1234, 0.1523]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", [])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_roys_indep(pillai_data):
    expected_stats = [0.0016, 0.0007, 0.0020, 0.0080, 0.0020]
    expected_pvalues = [0.2125, 0.4154, 0.5741, 0.0452, 0.5741]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_roys_dependent(pillai_data):
    expected_stats = [0.1700, 0.2159, 0.1717, 0.1428, 0.1717]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


def test_roys_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", [])
        assert test.statistic_ >= 0.05
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"][:3] + pillai_data["indep"][4:]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ <= 0.1
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ >= 0.05
        assert test.p_value_ <= 0.05
