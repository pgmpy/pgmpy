import numpy as np
import pytest

from pgmpy.ci_tests import WilksLambda
from pgmpy.tests.test_ci_tests import _multivariate_fixtures

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_wilks_no_cond(pillai_data):
    expected_stats = [0.7985, 0.8335, 0.8816, 0.8962, 0.8816]
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
    expected_stats = [0.9981, 0.9993, 0.9989, 0.9957, 0.9989]
    expected_pvalues = [0.1736, 0.3919, 0.5894, 0.3686, 0.5894]

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
    expected_stats = [0.7871, 0.7971, 0.9004, 0.8592, 0.9004]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


def test_effect_size(pillai_data):
    expected_indep = [0.0019, 0.0007, 0.0011, 0.0021, 0.0011]
    expected_dep = [0.2129, 0.2029, 0.0996, 0.0731, 0.0996]

    for df, expected in zip(pillai_data["indep"], expected_indep):
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)

    for df, expected in zip(pillai_data["dep"], expected_dep):
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)


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
