import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import ChiSquare


@pytest.fixture
def test_chi_square():
    df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    test = ChiSquare(data=df_adult)

    return test


def test_chi_square_effect_size_strong_association():
    """Cramér's V should be large for a strong association and ~0 under independence."""
    rng = np.random.RandomState(42)
    n = 5000
    a = rng.randint(0, 3, size=n)
    df = pd.DataFrame({"A": a, "B": a, "C": rng.randint(0, 3, size=n)}, dtype=str)
    test = ChiSquare(data=df)

    test("A", "B", Z=[], significance_level=0.05)
    # A == B is a perfect 3x3 association: V should be 1.0
    assert test.effect_size_ == pytest.approx(1.0, abs=0.01)
    # Hand-check formula: V = sqrt(chi / (n * (k_min - 1))).
    expected_v = float(np.sqrt(test.statistic_ / (n * 2)))
    assert test.effect_size_ == pytest.approx(expected_v, abs=1e-6)

    test("A", "C", Z=[], significance_level=0.05)
    # A and C are independent — Cramér's V should be tiny.
    assert test.effect_size_ < 0.05


def test_chi_square_effect_size_within_unit_interval():
    """Cramér's V is always in [0, 1] regardless of conditioning structure."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randint(0, 4, size=(2000, 5)), columns=list("ABCDE"), dtype=str)
    test = ChiSquare(data=df)
    for X, Y, Z in [("A", "B", []), ("A", "B", ["C"]), ("A", "B", ["C", "D"])]:
        test(X, Y, Z=Z, significance_level=0.05)
        assert 0.0 <= test.effect_size_ <= 1.0


def test_chisquare_adult_dataset(test_chi_square):
    # Comparison values taken from dagitty (DAGitty)
    test_chi_square("Age", "Immigrant", [])
    assert test_chi_square.statistic_ == pytest.approx(57.75, abs=0.1)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-25.47, abs=0.1)
    assert test_chi_square.dof_ == 4

    test_chi_square("Age", "Race", [])
    assert test_chi_square.statistic_ == pytest.approx(56.25, abs=0.1)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-24.75, abs=0.1)
    assert test_chi_square.dof_ == 4

    test_chi_square("Age", "Sex", [])
    assert test_chi_square.statistic_ == pytest.approx(289.62, abs=0.1)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-139.82, abs=0.1)
    assert test_chi_square.dof_ == 4

    test_chi_square(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
    )
    assert test_chi_square.statistic_ == pytest.approx(1460.11, abs=0.1)
    assert test_chi_square.p_value_ == pytest.approx(0, abs=0.1)
    assert test_chi_square.dof_ == 316

    test_chi_square("Immigrant", "Sex", [])
    assert test_chi_square.statistic_ == pytest.approx(0.2724, abs=0.1)
    assert np.log(test_chi_square.p_value_) == pytest.approx(-0.50, abs=0.1)
    assert test_chi_square.dof_ == 1

    test_chi_square("Education", "MaritalStatus", ["Age", "Sex"])
    assert test_chi_square.statistic_ == pytest.approx(481.96, abs=0.1)
    assert test_chi_square.p_value_ == pytest.approx(0, abs=0.1)
    assert test_chi_square.dof_ == 58

    # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
    # dataframes with very few samples. Update: Might be same from scipy=1.7.0
    test_chi_square(
        "Income",
        "Race",
        ["Age", "Education", "HoursPerWeek", "MaritalStatus"],
    )
    assert test_chi_square.statistic_ == pytest.approx(66.39, abs=0.1)
    assert test_chi_square.p_value_ == pytest.approx(0.99, abs=0.1)
    assert test_chi_square.dof_ == 136

    test_chi_square(
        "Immigrant",
        "Income",
        ["Age", "Education", "HoursPerWeek", "MaritalStatus"],
    )
    assert test_chi_square.statistic_ == pytest.approx(65.59, abs=0.1)
    assert test_chi_square.p_value_ == pytest.approx(0.999, abs=0.01)
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


def test_exactly_same_vars():
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    test = ChiSquare(data=df)
    test("x", "y", [])
    assert test.dof_ == 1
    assert test.p_value_ == pytest.approx(0, abs=1e-5)
