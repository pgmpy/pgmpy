import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import GSq


@pytest.fixture
def test_data():
    df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    test = GSq(data=df_adult)

    return test


def test_discrete_tests(test_data):
    assert not test_data("Age", "Immigrant", [], significance_level=0.05)
    assert not test_data("Age", "Race", [], significance_level=0.05)
    assert not test_data("Age", "Sex", [], significance_level=0.05)
    assert not test_data(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
        significance_level=0.05,
    )
    assert test_data("Immigrant", "Sex", [], significance_level=0.05)
    assert not test_data("Education", "MaritalStatus", ["Age", "Sex"], significance_level=0.05)


def test_exactly_same_vars():
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    test = GSq(data=df)
    test("x", "y", [])
    assert test.dof_ == 1
    assert test.p_value_ == pytest.approx(0, abs=1e-2)
