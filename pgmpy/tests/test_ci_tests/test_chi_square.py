import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import ChiSquare


@pytest.fixture
def test_chi_square():
    df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    test = ChiSquare(data=df_adult)

    return test


def test_chisquare_adult_dataset(test_chi_square):
    # Comparison values taken from dagitty (DAGitty)
    test_chi_square("Age", "Immigrant", [])
    assert test_chi_square.statistic_ == pytest.approx(57.75, abs=1e-2)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-25.47, abs=1e-2)
    assert test_chi_square.dof_ == 4

    test_chi_square("Age", "Race", [])
    assert test_chi_square.statistic_ == pytest.approx(56.25, abs=1e-2)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-24.75, abs=1e-2)
    assert test_chi_square.dof_ == 4

    test_chi_square("Age", "Sex", [])
    assert test_chi_square.statistic_ == pytest.approx(289.62, abs=1e-2)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-139.82, abs=1e-2)
    assert test_chi_square.dof_ == 4

    test_chi_square(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
    )
    assert test_chi_square.statistic_ == pytest.approx(1460.11, abs=1e-2)
    assert test_chi_square.p_value_ == pytest.approx(0, abs=1e-2)
    assert test_chi_square.dof_ == 316

    test_chi_square("Immigrant", "Sex", [])
    assert test_chi_square.statistic_ == pytest.approx(0.2503, abs=1e-2)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-0.48, abs=1e-2)
    assert test_chi_square.dof_ == 1

    test_chi_square("Education", "MaritalStatus", ["Age", "Sex"])
    assert test_chi_square.statistic_ == pytest.approx(481.96, abs=1e-2)
    assert test_chi_square.p_value_ == pytest.approx(0, abs=1e-2)
    assert test_chi_square.dof_ == 58

    # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
    # dataframes with very few samples. Update: Might be same from scipy=1.7.0
    test_chi_square(
        "Income",
        "Race",
        ["Age", "Education", "HoursPerWeek", "MaritalStatus"],
    )
    assert test_chi_square.statistic_ == pytest.approx(66.39, abs=1e-2)
    assert test_chi_square.p_value_ == pytest.approx(0.99, abs=1e-2)
    assert test_chi_square.dof_ == 136

    test_chi_square(
        "Immigrant",
        "Income",
        ["Age", "Education", "HoursPerWeek", "MaritalStatus"],
    )
    assert test_chi_square.statistic_ == pytest.approx(65.59, abs=1e-2)
    assert test_chi_square.p_value_ == pytest.approx(0.999, abs=1e-2)
    assert test_chi_square.dof_ == 131


def test_discrete_tests(test_chi_square):
    assert not test_chi_square("Age", "Immigrant", [], significance_level=0.05)
    assert not test_chi_square("Age", "Race", [], significance_level=0.05)
    assert not test_chi_square("Age", "Sex", [], significance_level=0.05)
    assert not test_chi_square(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
        significance_level=0.05,
    )
    assert test_chi_square("Immigrant", "Sex", [], significance_level=0.05)
    assert not test_chi_square("Education", "MaritalStatus", ["Age", "Sex"], significance_level=0.05)


def test_effect_size(test_chi_square):
    test_chi_square("Age", "Immigrant", [])
    assert test_chi_square.effect_size_ == pytest.approx(0.0421, abs=1e-2)

    test_chi_square("Immigrant", "Sex", [])
    assert test_chi_square.effect_size_ == pytest.approx(0.0029, abs=1e-2)

    test_chi_square("Education", "MaritalStatus", ["Age", "Sex"])
    assert test_chi_square.effect_size_ == pytest.approx(0.0894, abs=1e-2)


def test_exactly_same_vars():
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    test = ChiSquare(data=df)
    test("x", "y", [])
    assert test.dof_ == 1
    assert test.p_value_ == pytest.approx(0, abs=1e-2)
