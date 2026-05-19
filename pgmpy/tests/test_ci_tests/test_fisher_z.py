import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pgmpy.ci_tests import FisherZ, Pearsonr
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@pytest.fixture
def fisher_data():
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
    Z = 0.2 * X + 0.2 * Y + rng.normal(loc=0, scale=0.1, size=10000)
    df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    return df_ind, df_cind, df_cind_mul, df_vstruct


def test_fisher_z_basic(fisher_data):
    df_ind, df_cind, df_cind_mul, df_vstruct = fisher_data
    test = FisherZ(data=df_ind)
    test("X", "Y", [])
    assert test.p_value_ > 0.05

    test = FisherZ(data=df_cind)
    test("X", "Y", ["Z"])
    assert test.p_value_ > 0.05

    test = FisherZ(data=df_cind_mul)
    test("X", "Y", ["Z1", "Z2"])
    assert test.p_value_ > 0.05

    test = FisherZ(data=df_vstruct)
    test("X", "Y", ["Z"])
    assert test.p_value_ < 0.05

    assert FisherZ(data=df_ind)("X", "Y", [], significance_level=0.05)
    assert FisherZ(data=df_cind)("X", "Y", ["Z"], significance_level=0.05)
    assert FisherZ(data=df_cind_mul)("X", "Y", ["Z1", "Z2"], significance_level=0.05)
    assert not FisherZ(data=df_vstruct)("X", "Y", ["Z"], significance_level=0.05)


def test_fisher_z_uses_pearsonr_partial_correlation():
    rng = np.random.default_rng(seed=7)
    n_samples = 40
    z_columns = [f"Z{i}" for i in range(5)]
    data = pd.DataFrame(rng.standard_normal((n_samples, len(z_columns))), columns=z_columns)
    data["X"] = data[z_columns].sum(axis=1) + rng.normal(scale=1.0, size=n_samples)
    data["Y"] = 0.35 * data["X"] + data[z_columns].sum(axis=1) + rng.normal(scale=1.2, size=n_samples)

    pearson_test = Pearsonr(data=data)
    pearson_test("X", "Y", z_columns)

    fisher_test = FisherZ(data=data)
    is_independent = fisher_test("X", "Y", z_columns, significance_level=0.05)

    expected_coeff = np.arctanh(np.clip(pearson_test.statistic_, -0.999999, 0.999999))
    expected_statistic = np.sqrt(n_samples - len(z_columns) - 3) * expected_coeff
    expected_p_value = 2 * stats.norm.sf(np.abs(expected_statistic))

    assert is_independent == (expected_p_value >= 0.05)
    assert fisher_test.statistic_ == pytest.approx(expected_statistic, abs=1e-2)
    assert fisher_test.p_value_ == pytest.approx(expected_p_value, abs=1e-2)


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
def test_fisher_z_residual(residual_data):
    df_indep, df_dep = residual_data
    test = FisherZ(data=df_indep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.p_value_, 3) == pytest.approx(0.044, abs=1e-2)

    test = FisherZ(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.p_value_, 3) == pytest.approx(0.0, abs=1e-2)


def test_effect_size(fisher_data):
    df_ind, df_cind, df_cind_mul, df_vstruct = fisher_data

    test = FisherZ(data=df_ind)
    test("X", "Y", [])
    assert test.effect_size_ == pytest.approx(0.0075, abs=1e-2)

    test = FisherZ(data=df_vstruct)
    test("X", "Y", ["Z"])
    assert test.effect_size_ == pytest.approx(0.8010, abs=1e-2)

    pearson_test = Pearsonr(data=df_vstruct)
    pearson_test("X", "Y", ["Z"])
    assert test.effect_size_ == pytest.approx(abs(pearson_test.statistic_), abs=1e-2)


def test_fisher_z_residual_approx(residual_data):
    df_indep, df_dep = residual_data
    test = FisherZ(data=df_indep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.p_value_ >= 0.03

    test = FisherZ(data=df_dep)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.p_value_ <= 0.05
