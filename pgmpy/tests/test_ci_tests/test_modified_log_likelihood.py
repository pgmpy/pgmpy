import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import ModifiedLogLikelihood


@pytest.fixture
def test_mll():
    df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
    test = ModifiedLogLikelihood(data=df_adult)

    return test


def test_discrete_tests(test_mll):
    assert not test_mll("Age", "Immigrant", [], significance_level=0.05)
    assert not test_mll("Age", "Race", [], significance_level=0.05)
    assert not test_mll("Age", "Sex", [], significance_level=0.05)
    assert not test_mll(
        "Education",
        "HoursPerWeek",
        ["Age", "Immigrant", "Race", "Sex"],
        significance_level=0.05,
    )

    assert test_mll("Immigrant", "Sex", [], significance_level=0.05)
    assert not test_mll("Education", "MaritalStatus", ["Age", "Sex"], significance_level=0.05)


def test_exactly_same_vars():
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    test = ModifiedLogLikelihood(data=df)
    test("x", "y", [])
    assert test.dof_ == 1
    assert test.p_value_ == pytest.approx(0, abs=1e-2)
