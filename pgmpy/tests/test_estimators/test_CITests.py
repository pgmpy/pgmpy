import os

import numpy as np
import pandas as pd
import pytest
from numpy import testing as np_test
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.estimators.CITests import (
    chi_square,
    ci_registry,
    g_sq,
    gcm,
    log_likelihood,
    modified_log_likelihood,
    pearsonr,
    pearsonr_equivalence,
    pillai_trace,
)
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestCIRegistry:
    def test_ci_registry(self):
        all_tests = ci_registry.list_all()

        assert "chi_square" in all_tests
        assert "g_sq" in all_tests
        assert "log_likelihood" in all_tests
        assert "modified_log_likelihood" in all_tests
        assert "pearsonr" in all_tests
        assert "pillai" in all_tests
        assert "gcm" in all_tests


@pytest.fixture
def pearsonr_data():
    rng = np.random.default_rng(seed=42)

    df_ind = pd.DataFrame(np.random.randn(1000, 3), columns=["X", "Y", "Z"])

    Z = rng.normal(10000)
    X = 3 * Z + rng.normal(loc=0, scale=0.1, size=10000)
    Y = 2 * Z + rng.normal(loc=0, scale=0.1, size=10000)

    df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    Z1 = rng.normal(10000)
    Z2 = rng.normal(10000)
    X = 3 * Z1 + 2 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
    Y = 2 * Z1 + 3 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
    df_cind_mul = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

    X = rng.normal(10000)
    Y = rng.normal(10000)
    Z = 2 * X + 2 * Y + rng.normal(loc=0, scale=0.1, size=10000)
    df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    return {
        "df_ind": df_ind,
        "df_cind": df_cind,
        "df_cind_mul": df_cind_mul,
        "df_vstruct": df_vstruct,
    }


class TestPearsonr:
    def test_pearsonr(self, pearsonr_data):
        df_ind = pearsonr_data["df_ind"]
        df_cind = pearsonr_data["df_cind"]
        df_cind_mul = pearsonr_data["df_cind_mul"]
        df_vstruct = pearsonr_data["df_vstruct"]

        coef, p_value = pearsonr(X="X", Y="Y", Z=[], data=df_ind, boolean=False)
        assert coef < 0.1
        assert p_value > 0.05

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z"], data=df_cind, boolean=False
        )
        assert coef < 0.1
        assert p_value > 0.05

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z1", "Z2"], data=df_cind_mul, boolean=False
        )
        assert coef < 0.1
        assert p_value > 0.05

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z"], data=df_vstruct, boolean=False
        )
        assert abs(coef) > 0.9
        assert p_value < 0.05

        # Tests for when boolean=True
        assert pearsonr(X="X", Y="Y", Z=[], data=df_ind, significance_level=0.05)
        assert pearsonr(X="X", Y="Y", Z=["Z"], data=df_cind, significance_level=0.05)
        assert pearsonr(
            X="X",
            Y="Y",
            Z=["Z1", "Z2"],
            data=df_cind_mul,
            significance_level=0.05,
        )
        assert not pearsonr(
            X="X", Y="Y", Z=["Z"], data=df_vstruct, significance_level=0.05
        )


@pytest.fixture
def adult_dataset():
    return pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")


class TestDiscreteTests:
    def test_chisquare_adult_dataset(self, adult_dataset):
        df_adult = adult_dataset
        # Comparison values taken from dagitty (DAGitty)
        coef, p_value, dof = chi_square(
            X="Age", Y="Immigrant", Z=[], data=df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 57.75, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -25.47, decimal=1)
        assert dof == 4

        coef, p_value, dof = chi_square(
            X="Age", Y="Race", Z=[], data=df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 56.25, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -24.75, decimal=1)
        assert dof == 4

        coef, p_value, dof = chi_square(
            X="Age", Y="Sex", Z=[], data=df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 289.62, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -139.82, decimal=1)
        assert dof == 4

        coef, p_value, dof = chi_square(
            X="Education",
            Y="HoursPerWeek",
            Z=["Age", "Immigrant", "Race", "Sex"],
            data=df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 1460.11, decimal=1)
        np_test.assert_almost_equal(p_value, 0, decimal=1)
        assert dof == 316

        coef, p_value, dof = chi_square(
            X="Immigrant", Y="Sex", Z=[], data=df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 0.2724, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -0.50, decimal=1)
        assert dof == 1

        coef, p_value, dof = chi_square(
            X="Education",
            Y="MaritalStatus",
            Z=["Age", "Sex"],
            data=df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 481.96, decimal=1)
        np_test.assert_almost_equal(p_value, 0, decimal=1)
        assert dof == 58

        # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
        # dataframes with very few samples. Update: Might be same from scipy=1.7.0
        coef, p_value, dof = chi_square(
            X="Income",
            Y="Race",
            Z=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
            data=df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 66.39, decimal=1)
        np_test.assert_almost_equal(p_value, 0.99, decimal=1)
        assert dof == 136

        coef, p_value, dof = chi_square(
            X="Immigrant",
            Y="Income",
            Z=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
            data=df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 65.59, decimal=1)
        np_test.assert_almost_equal(p_value, 0.999, decimal=2)
        assert dof == 131

    def test_discrete_tests(self, adult_dataset):
        df_adult = adult_dataset
        for t in [
            chi_square,
            g_sq,
            log_likelihood,
            modified_log_likelihood,
        ]:
            assert not t(
                X="Age",
                Y="Immigrant",
                Z=[],
                data=df_adult,
                boolean=True,
                significance_level=0.05,
            )

            assert not t(
                X="Age",
                Y="Race",
                Z=[],
                data=df_adult,
                boolean=True,
                significance_level=0.05,
            )

            assert not t(
                X="Age",
                Y="Sex",
                Z=[],
                data=df_adult,
                boolean=True,
                significance_level=0.05,
            )

            assert not t(
                X="Education",
                Y="HoursPerWeek",
                Z=["Age", "Immigrant", "Race", "Sex"],
                data=df_adult,
                boolean=True,
                significance_level=0.05,
            )
            assert t(
                X="Immigrant",
                Y="Sex",
                Z=[],
                data=df_adult,
                boolean=True,
                significance_level=0.05,
            )
            assert not t(
                X="Education",
                Y="MaritalStatus",
                Z=["Age", "Sex"],
                data=df_adult,
                boolean=True,
                significance_level=0.05,
            )

    def test_exactly_same_vars(self):
        x = np.random.choice([0, 1], size=1000)
        y = x.copy()
        df = pd.DataFrame({"x": x, "y": y})

        for t in [
            chi_square,
            g_sq,
            log_likelihood,
            modified_log_likelihood,
        ]:
            stat, p_value, dof = t(X="x", Y="y", Z=[], data=df, boolean=False)
            assert dof == 1
            np_test.assert_almost_equal(p_value, 0, decimal=5)


@pytest.fixture
def residual_methods_data():
    # Create a combination of mixed data types
    np.random.seed(42)

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
    cpd_y_indep = LinearGaussianCPD(
        "Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"]
    )
    model_indep.add_cpds(
        cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_indep
    )
    df_indep = model_indep.simulate(n_samples=1000, seed=42)

    df_indep_cont_cont = df_indep.copy()
    df_indep_cont_cont.Z2 = pd.cut(
        df_indep_cont_cont.Z2,
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    )

    df_indep_cat_cont = df_indep_cont_cont.copy()
    df_indep_cat_cont.X = pd.cut(
        df_indep_cat_cont.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    )

    df_indep_cat_cat = df_indep_cont_cont.copy()
    df_indep_cat_cat.X = pd.cut(
        df_indep_cat_cat.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    )
    df_indep_cat_cat.Y = pd.cut(
        df_indep_cat_cat.Y,
        bins=4,
        ordered=False,
        labels=["y1", "y2", "y3", "y4"],
    )

    df_indep_ord_cont = df_indep_cont_cont.copy()
    df_indep_ord_cont.X = pd.cut(df_indep_ord_cont.X, bins=4)

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
    cpd_y_dep = LinearGaussianCPD(
        "Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"]
    )
    model_dep.add_cpds(
        cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep
    )
    df_dep = model_dep.simulate(n_samples=1000, seed=42)

    df_dep_cont_cont = df_dep.copy()
    df_dep_cont_cont.Z2 = pd.cut(
        df_dep_cont_cont.Z2,
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    )

    df_dep_cat_cont = df_dep_cont_cont.copy()
    df_dep_cat_cont.X = pd.cut(
        df_dep_cat_cont.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    )

    df_dep_cat_cat = df_dep_cont_cont.copy()
    df_dep_cat_cat.X = pd.cut(
        df_dep_cat_cat.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    )
    df_dep_cat_cat.Y = pd.cut(
        df_dep_cat_cat.Y,
        bins=4,
        ordered=False,
        labels=["y1", "y2", "y3", "y4"],
    )

    df_dep_ord_cont = df_dep_cont_cont.copy()
    df_dep_ord_cont.X = pd.cut(df_dep_ord_cont.X, bins=4)

    return {
        "df_indep": df_indep,
        "df_indep_cont_cont": df_indep_cont_cont,
        "df_indep_cat_cont": df_indep_cat_cont,
        "df_indep_cat_cat": df_indep_cat_cat,
        "df_indep_ord_cont": df_indep_ord_cont,
        "df_dep": df_dep,
        "df_dep_cont_cont": df_dep_cont_cont,
        "df_dep_cat_cont": df_dep_cat_cont,
        "df_dep_cat_cat": df_dep_cat_cat,
        "df_dep_ord_cont": df_dep_ord_cont,
    }


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping residual tests on GitHub Actions."
)
class TestResidualMethods:
    def test_pearsonr(self, residual_methods_data):
        df_indep = residual_methods_data["df_indep"]
        df_dep = residual_methods_data["df_dep"]

        coef, p_value = pearsonr(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=df_indep,
            boolean=False,
            seed=42,
        )
        assert abs(coef) <= 0.1
        assert p_value >= 0.04

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=df_dep, boolean=False, seed=42
        )
        assert coef >= 0.1
        assert np.isclose(p_value, 0, atol=1e-1)

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_no_cond(self, residual_methods_data):
        dep_coefs = [0.2038, 0.2038, 0.1733, 0.1527, 0.1733]
        dep_pvalues = [0, 0, 0, 0, 0]

        computed_coefs = []
        computed_pvalues = []
        for i, df_indep in enumerate(
            [
                residual_methods_data["df_indep"],
                residual_methods_data["df_indep_cont_cont"],
                residual_methods_data["df_indep_cat_cont"],
                residual_methods_data["df_indep_cat_cat"],
                residual_methods_data["df_indep_ord_cont"],
            ]
        ):
            coef, p_value = pillai_trace(
                X="X",
                Y="Y",
                Z=[],
                data=df_indep,
                boolean=False,
                seed=42,
            )
            computed_coefs.append(coef)
            computed_pvalues.append(p_value)

        assert np.allclose(computed_coefs, dep_coefs, rtol=1e-2, atol=1e-2), (
            f"Non-conditional coefs mismatch at index {i}: {computed_coefs} != {dep_coefs}"
        )
        assert np.allclose(computed_pvalues, dep_pvalues, rtol=1e-2, atol=1e-2), (
            f"Non-conditional p-values mismatch at index {i}: {computed_pvalues} != {dep_pvalues}"
        )

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_indep(self, residual_methods_data):
        indep_coefs = [0.0014, 0.0023, 0.0041, 0.0213, 0.0041]
        indep_pvalues = [0.2430, 0.0161, 0.0522, 0.0184, 0.0522]

        computed_coefs = []
        computed_pvalues = []
        for i, df_indep in enumerate(
            [
                residual_methods_data["df_indep"],
                residual_methods_data["df_indep_cont_cont"],
                residual_methods_data["df_indep_cat_cont"],
                residual_methods_data["df_indep_cat_cat"],
                residual_methods_data["df_indep_ord_cont"],
            ]
        ):
            coef, p_value = pillai_trace(
                X="X",
                Y="Y",
                Z=["Z1", "Z2", "Z3"],
                data=df_indep,
                boolean=False,
                seed=42,
            )
            computed_coefs.append(coef)
            computed_pvalues.append(p_value)

        assert np.allclose(computed_coefs, indep_coefs, rtol=1e-2, atol=1e-2), (
            f"Conditional (indep) coefs mismatch at index {i}: {computed_coefs} != {indep_coefs}"
        )
        assert np.allclose(computed_pvalues, indep_pvalues, rtol=1e-2, atol=1e-2), (
            f"Conditional (indep) p-values mismatch at index {i}: {computed_pvalues} != {indep_pvalues}"
        )

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_dependent(self, residual_methods_data):
        dep_coefs = np.array([0.1322, 0.1609, 0.1182, 0.1330, 0.1182])
        dep_pvalues = np.array([0, 0, 0, 0, 0])

        computed_coefs = []
        computed_pvalues = []
        for i, df_dep in enumerate(
            [
                residual_methods_data["df_dep"],
                residual_methods_data["df_dep_cont_cont"],
                residual_methods_data["df_dep_cat_cont"],
                residual_methods_data["df_dep_cat_cat"],
                residual_methods_data["df_dep_ord_cont"],
            ]
        ):
            coef, p_value = pillai_trace(
                X="X",
                Y="Y",
                Z=["Z1", "Z2", "Z3"],
                data=df_dep,
                boolean=False,
                seed=42,
            )
            computed_coefs.append(coef)
            computed_pvalues.append(p_value)

        assert np.allclose(computed_coefs, dep_coefs, rtol=1e-2, atol=1e-2), (
            f"Conditional (dep) coefs mismatch at index {i}: {computed_coefs} != {dep_coefs}"
        )
        assert np.allclose(computed_pvalues, dep_pvalues, rtol=1e-2, atol=1e-2), (
            f"Conditional (dep) p-values mismatch at index {i}: {computed_pvalues} != {dep_pvalues}"
        )

    def test_gcm(self, residual_methods_data):
        df_indep = residual_methods_data["df_indep"]
        df_dep = residual_methods_data["df_dep"]

        # Non-conditional tests
        coef, p_value = gcm(
            X="X",
            Y="Y",
            Z=[],
            data=df_indep,
            boolean=False,
            seed=42,
        )
        assert round(coef, 3) == 13.693
        assert p_value == 0.0

        # Conditional tests
        coef, p_value = gcm(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=df_indep,
            boolean=False,
            seed=42,
        )

        assert round(coef, 3) == 0.097
        assert round(p_value, 4) == 0.9228

        # Conditional tests
        coef, p_value = gcm(
            X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=df_dep, boolean=False, seed=42
        )

        assert round(coef, 3) == 11.69
        assert p_value == 0.0

    def test_pearsonr_equivalence(self, residual_methods_data):
        df_dep = residual_methods_data["df_dep"]

        is_independent = pearsonr_equivalence(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=df_dep,
            boolean=True,
            significance_level=0.05,
            delta_th=0.3,
        )
        assert not is_independent

        is_independent = pearsonr_equivalence(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=df_dep,
            boolean=False,
            significance_level=0.05,
            delta_th=0.5,
        )
        assert is_independent
