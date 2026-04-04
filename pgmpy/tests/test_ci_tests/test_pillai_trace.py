import os

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.ci_tests import PillaiTrace
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

skip_ci = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping residual tests on GitHub Actions.",
)

skip_xgb = pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="requires xgboost",
)


@pytest.fixture
def pillai_data():
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

    df_indep_cont_cont = df_indep.copy()
    df_indep_cont_cont.Z2 = pd.cut(
        df_indep_cont_cont.Z2,
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    ).cat.codes

    df_indep_cat_cont = df_indep_cont_cont.copy()
    df_indep_cat_cont.X = pd.cut(
        df_indep_cat_cont.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes

    df_indep_cat_cat = df_indep_cont_cont.copy()
    df_indep_cat_cat.X = pd.cut(
        df_indep_cat_cat.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes
    df_indep_cat_cat.Y = pd.cut(
        df_indep_cat_cat.Y,
        bins=4,
        ordered=False,
        labels=["y1", "y2", "y3", "y4"],
    ).cat.codes

    df_indep_ord_cont = df_indep_cont_cont.copy()
    df_indep_ord_cont.X = pd.cut(df_indep_ord_cont.X, bins=4).cat.codes

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

    df_dep_cont_cont = df_dep.copy()
    df_dep_cont_cont.Z2 = pd.cut(
        df_dep_cont_cont.Z2,
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    ).cat.codes

    df_dep_cat_cont = df_dep_cont_cont.copy()
    df_dep_cat_cont.X = pd.cut(
        df_dep_cat_cont.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes

    df_dep_cat_cat = df_dep_cont_cont.copy()
    df_dep_cat_cat.X = pd.cut(
        df_dep_cat_cat.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes
    df_dep_cat_cat.Y = pd.cut(
        df_dep_cat_cat.Y,
        bins=4,
        ordered=False,
        labels=["y1", "y2", "y3", "y4"],
    ).cat.codes

    df_dep_ord_cont = df_dep_cont_cont.copy()
    df_dep_ord_cont.X = pd.cut(df_dep_ord_cont.X, bins=4).cat.codes

    return {
        "indep": [df_indep, df_indep_cont_cont, df_indep_cat_cont, df_indep_cat_cat, df_indep_ord_cont],
        "dep": [df_dep, df_dep_cont_cont, df_dep_cat_cont, df_dep_cat_cat, df_dep_ord_cont],
    }


@skip_xgb
@skip_ci
def test_pillai_no_cond(pillai_data):
    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df, seed=42)
        test("X", "Y", [])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.1)
    assert np.all(np.array(computed_pvalues) <= 0.05)


@skip_xgb
@skip_ci
def test_pillai_indep(pillai_data):
    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df, seed=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) <= 0.1)
    assert np.all(np.array(computed_pvalues) >= 0.05)


@skip_xgb
@skip_ci
def test_pillai_dependent(pillai_data):
    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["dep"]:
        test = PillaiTrace(data=df, seed=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.09)
    assert np.all(np.array(computed_pvalues) <= 0.05)


@skip_xgb
def test_pillai_tests_approx(pillai_data):
    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df, seed=42)
        test("X", "Y", [])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.1)
    assert np.all(np.array(computed_pvalues) <= 0.05)

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df, seed=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) <= 0.1)
    assert np.all(np.array(computed_pvalues) >= 0.05)

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["dep"]:
        test = PillaiTrace(data=df, seed=42)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.05)
    assert np.all(np.array(computed_pvalues) <= 0.05)
