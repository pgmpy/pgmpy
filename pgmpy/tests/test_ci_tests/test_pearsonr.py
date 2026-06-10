import os

import numpy as np
import pandas as pd
import pytest

from pgmpy.ci_tests import Pearsonr
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@pytest.fixture
def pearsonr_data():
    rng = np.random.default_rng(seed=42)

    df_ind = pd.DataFrame(rng.standard_normal((10000, 3)), columns=["X", "Y", "Z"])

    Z = rng.normal(size=10000)
    X = 0.3 * Z + rng.normal(loc=0, scale=0.1, size=10000)
    Y = 0.2 * Z + rng.normal(loc=0, scale=0.1, size=10000)
    df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    Z1 = rng.normal(size=10000)
    Z2 = rng.normal(size=10000)
    X = 0.3 * Z1 + 0.2 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
    Y = 0.2 * Z1 + 0.3 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
    df_cind_mul = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

    X = rng.normal(size=10000)
    Y = rng.normal(size=10000)
    Z = 0.8 * X + 0.8 * Y + rng.normal(loc=0, scale=0.1, size=10000)
    df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    return df_ind, df_cind, df_cind_mul, df_vstruct


def test_pearsonr(pearsonr_data):
    df_ind, df_cind, df_cind_mul, df_vstruct = pearsonr_data
    test = Pearsonr(data=df_ind)
    test("X", "Y", [])
    assert test.statistic_ < 0.1
    assert test.p_value_ > 0.05

    test = Pearsonr(data=df_cind)
    test("X", "Y", ["Z"])
    assert test.statistic_ < 0.1
    assert test.p_value_ > 0.05

    test = Pearsonr(data=df_cind_mul)
    test("X", "Y", ["Z1", "Z2"])
    assert test.statistic_ < 0.1
    assert test.p_value_ > 0.05

    test = Pearsonr(data=df_vstruct)
    test("X", "Y", ["Z"])
    assert abs(test.statistic_) > 0.5
    assert test.p_value_ < 0.05

    assert Pearsonr(data=df_ind)("X", "Y", [], significance_level=0.05)
    assert Pearsonr(data=df_cind)("X", "Y", ["Z"], significance_level=0.05)
    assert Pearsonr(data=df_cind_mul)("X", "Y", ["Z1", "Z2"], significance_level=0.05)
    assert not Pearsonr(data=df_vstruct)("X", "Y", ["Z"], significance_level=0.05)


@pytest.fixture
def residual_data():
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
    df_indep = model_indep.simulate(n_samples=1000, seed=42)

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
    cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
    model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
    df_dep = model_dep.simulate(n_samples=1000, seed=42)

    return df_indep, df_dep


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping residual tests on GitHub Actions.",
)
def test_pearsonr_residual(residual_data):
    df_indep, df_dep = residual_data
    test = Pearsonr(data=df_indep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.statistic_, 3) == pytest.approx(-0.056, abs=1e-2)
    assert round(test.p_value_, 3) == pytest.approx(0.044, abs=1e-2)

    test = Pearsonr(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.statistic_, 3) == pytest.approx(0.406, abs=1e-2)
    assert round(test.p_value_, 2) == pytest.approx(0.0, abs=1e-2)


def test_effect_size(pearsonr_data):
    df_ind, df_cind, df_cind_mul, df_vstruct = pearsonr_data

    test = Pearsonr(data=df_ind)
    test("X", "Y", [])
    assert test.effect_size_ == pytest.approx(abs(test.statistic_), abs=1e-2)
    assert test.effect_size_ == pytest.approx(0.0075, abs=1e-2)

    test = Pearsonr(data=df_vstruct)
    test("X", "Y", ["Z"])
    assert test.effect_size_ == pytest.approx(abs(test.statistic_), abs=1e-2)
    assert test.effect_size_ == pytest.approx(0.9846, abs=1e-2)


def test_pearsonr_residual_approx(residual_data):
    df_indep, df_dep = residual_data
    test = Pearsonr(data=df_indep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert abs(test.statistic_) <= 0.1
    assert test.p_value_ >= 0.01

    test = Pearsonr(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.statistic_ >= 0.1
    assert test.p_value_ <= 0.05
