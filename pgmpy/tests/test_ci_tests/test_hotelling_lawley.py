import numpy as np

from pgmpy.ci_tests import HotellingLawley
from pgmpy.tests.test_ci_tests import _multivariate_fixtures

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_hotelling_no_cond(pillai_data):
    expected_stats = [0.1865, 0.1865, 0.1572, 0.1180, 0.1572]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = HotellingLawley(data=df)
        test("X", "Y", [])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_hotelling_indep(pillai_data):
    expected_stats = [0.0016, 0.0007, 0.0044, 0.0055, 0.0044]
    expected_pvalues = [0.2125, 0.4154, 0.1118, 0.2405, 0.1118]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = HotellingLawley(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_hotelling_dependent(pillai_data):
    expected_stats = [0.2049, 0.2754, 0.1542, 0.1745, 0.1542]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = HotellingLawley(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


def test_hotelling_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = HotellingLawley(data=df)
        test("X", "Y", [])
        assert test.statistic_ >= 0.05
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"]:
        test = HotellingLawley(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ <= 0.1
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = HotellingLawley(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ >= 0.05
        assert test.p_value_ <= 0.05
