import os

import pytest
from sklearn.ensemble import RandomForestRegressor

from pgmpy.ci_tests import GCM
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@pytest.fixture
def df_indep():
    model_indep = LinearGaussianBayesianNetwork(
        [
            ("Z1", "X"),
            ("Z2", "X"),
            ("Z3", "X"),
            ("Z1", "Y"),
            ("Z2", "Y"),
            ("Z3", "Y"),
        ]
    )
    cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
    cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
    cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
    cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    cpd_y_indep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    model_indep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_indep)
    df_indep = model_indep.simulate(n_samples=10000, seed=42)

    return df_indep


@pytest.fixture
def df_dep():
    model_dep = LinearGaussianBayesianNetwork(
        [
            ("Z1", "X"),
            ("Z2", "X"),
            ("Z3", "X"),
            ("Z1", "Y"),
            ("Z2", "Y"),
            ("Z3", "Y"),
            ("X", "Y"),
        ]
    )
    cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
    cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
    cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
    cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
    model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
    df_dep = model_dep.simulate(n_samples=10000, seed=42)

    return df_dep


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping exact residual tests on GitHub Actions.")
def test_gcm_exact(df_dep, df_indep):
    test = GCM(data=df_indep)

    # Non-conditional test
    test("X", "Y", [])
    assert round(test.statistic_, 3) == pytest.approx(38.858, abs=1e-2)
    assert test.p_value_ == pytest.approx(0.0, abs=1e-2)

    # Conditional test (independent)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.statistic_, 3) == pytest.approx(-0.357, abs=1e-2)
    assert round(test.p_value_, 4) == pytest.approx(0.7207, abs=1e-2)

    # Conditional test (dependent)
    test = GCM(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.statistic_, 3) == pytest.approx(39.798, abs=1e-2)
    assert test.p_value_ == pytest.approx(0.0, abs=1e-2)

    # Test with custom sklearn estimator
    test = GCM(data=df_indep, estimator=RandomForestRegressor(random_state=42))
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert isinstance(test.statistic_, float)
    assert isinstance(test.p_value_, float)
    assert test.p_value_ >= 0.0
    assert test.p_value_ <= 1.0


def test_effect_size(df_dep, df_indep):
    test = GCM(data=df_indep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.effect_size_ == pytest.approx(0.0031, abs=1e-2)

    test = GCM(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.effect_size_ == pytest.approx(0.4371, abs=1e-2)


def test_gcm_approx(df_dep, df_indep):
    test = GCM(data=df_indep)

    # Non-conditional test
    test("X", "Y", [])
    assert test.statistic_ > 1
    assert test.p_value_ < 0.05

    # Conditional test (independent)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.statistic_ < 1
    assert test.p_value_ > 0.05

    # Conditional test (dependent)
    test = GCM(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.statistic_ > 1
    assert test.p_value_ < 0.05

    # Test with custom sklearn estimator
    test = GCM(data=df_indep, estimator=RandomForestRegressor(random_state=42))
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.p_value_ > 0.05
