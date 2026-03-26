import os
import unittest

import numpy as np
import pandas as pd
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
    regression_based_lr,
)
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestCIRegistry(unittest.TestCase):
    def test_ci_registry(self):
        all_tests = ci_registry.list_all()

        self.assertIn("chi_square", all_tests)
        self.assertIn("g_sq", all_tests)
        self.assertIn("log_likelihood", all_tests)
        self.assertIn("modified_log_likelihood", all_tests)
        self.assertIn("pearsonr", all_tests)
        self.assertIn("pillai", all_tests)
        self.assertIn("gcm", all_tests)


class TestPearsonr(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42)

        self.df_ind = pd.DataFrame(rng.standard_normal(size=(1000, 3)), columns=["X", "Y", "Z"])

        Z = rng.normal(size=10000)
        X = 3 * Z + rng.normal(loc=0, scale=0.1, size=10000)
        Y = 2 * Z + rng.normal(loc=0, scale=0.1, size=10000)

        self.df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        Z1 = rng.normal(size=10000)
        Z2 = rng.normal(size=10000)
        X = 3 * Z1 + 2 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
        Y = 2 * Z1 + 3 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
        self.df_cind_mul = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

        X = rng.normal(size=10000)
        Y = rng.normal(size=10000)
        Z = 2 * X + 2 * Y + rng.normal(loc=0, scale=0.1, size=10000)
        self.df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_pearsonr(self):
        coef, p_value = pearsonr(X="X", Y="Y", Z=[], data=self.df_ind, boolean=False)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_cind, boolean=False)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z1", "Z2"], data=self.df_cind_mul, boolean=False)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_vstruct, boolean=False)
        self.assertTrue(abs(coef) > 0.9)
        self.assertTrue(p_value < 0.05)

        # Tests for when boolean=True
        self.assertTrue(pearsonr(X="X", Y="Y", Z=[], data=self.df_ind, significance_level=0.05))
        self.assertTrue(pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_cind, significance_level=0.05))
        self.assertTrue(
            pearsonr(
                X="X",
                Y="Y",
                Z=["Z1", "Z2"],
                data=self.df_cind_mul,
                significance_level=0.05,
            )
        )
        self.assertFalse(pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_vstruct, significance_level=0.05))


class TestDiscreteTests(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    def test_chisquare_adult_dataset(self):
        # Comparison values taken from dagitty (DAGitty)
        coef, p_value, dof = chi_square(X="Age", Y="Immigrant", Z=[], data=self.df_adult, boolean=False)
        np_test.assert_almost_equal(coef, 57.75, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -25.47, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(X="Age", Y="Race", Z=[], data=self.df_adult, boolean=False)
        np_test.assert_almost_equal(coef, 56.25, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -24.75, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(X="Age", Y="Sex", Z=[], data=self.df_adult, boolean=False)
        np_test.assert_almost_equal(coef, 289.62, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -139.82, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(
            X="Education",
            Y="HoursPerWeek",
            Z=["Age", "Immigrant", "Race", "Sex"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 1460.11, decimal=1)
        np_test.assert_almost_equal(p_value, 0, decimal=1)
        self.assertEqual(dof, 316)

        coef, p_value, dof = chi_square(X="Immigrant", Y="Sex", Z=[], data=self.df_adult, boolean=False)
        np_test.assert_almost_equal(coef, 0.2724, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -0.50, decimal=1)
        self.assertEqual(dof, 1)

        coef, p_value, dof = chi_square(
            X="Education",
            Y="MaritalStatus",
            Z=["Age", "Sex"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 481.96, decimal=1)
        np_test.assert_almost_equal(p_value, 0, decimal=1)
        self.assertEqual(dof, 58)

        # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
        # dataframes with very few samples. Update: Might be same from scipy=1.7.0
        coef, p_value, dof = chi_square(
            X="Income",
            Y="Race",
            Z=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 66.39, decimal=1)
        np_test.assert_almost_equal(p_value, 0.99, decimal=1)
        self.assertEqual(dof, 136)

        coef, p_value, dof = chi_square(
            X="Immigrant",
            Y="Income",
            Z=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 65.59, decimal=1)
        np_test.assert_almost_equal(p_value, 0.999, decimal=2)
        self.assertEqual(dof, 131)

    def test_discrete_tests(self):
        for t in [
            chi_square,
            g_sq,
            log_likelihood,
            modified_log_likelihood,
        ]:
            self.assertFalse(
                t(
                    X="Age",
                    Y="Immigrant",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

            self.assertFalse(
                t(
                    X="Age",
                    Y="Race",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

            self.assertFalse(
                t(
                    X="Age",
                    Y="Sex",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )

            self.assertFalse(
                t(
                    X="Education",
                    Y="HoursPerWeek",
                    Z=["Age", "Immigrant", "Race", "Sex"],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )
            self.assertTrue(
                t(
                    X="Immigrant",
                    Y="Sex",
                    Z=[],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
            )
            self.assertFalse(
                t(
                    X="Education",
                    Y="MaritalStatus",
                    Z=["Age", "Sex"],
                    data=self.df_adult,
                    boolean=True,
                    significance_level=0.05,
                )
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
            self.assertEqual(dof, 1)
            np_test.assert_almost_equal(p_value, 0, decimal=5)


@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping residual tests on GitHub Actions.")
class TestResidualMethods(unittest.TestCase):
    def setUp(self):
        # Create a combination of mixed data types
        np.random.seed(42)

        self.model_indep = LinearGaussianBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
            ]
        )
        self.cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
        self.cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
        self.cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
        self.cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        self.cpd_y_indep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        self.model_indep.add_cpds(self.cpd_z1, self.cpd_z2, self.cpd_z3, self.cpd_x, self.cpd_y_indep)
        self.df_indep = self.model_indep.simulate(n_samples=1000, seed=42)

        self.df_indep_cont_cont = self.df_indep.copy()
        self.df_indep_cont_cont.Z2 = pd.cut(
            self.df_indep_cont_cont.Z2,
            bins=4,
            ordered=False,
            labels=["z21", "z22", "z23", "z24"],
        )

        self.df_indep_cat_cont = self.df_indep_cont_cont.copy()
        self.df_indep_cat_cont.X = pd.cut(
            self.df_indep_cat_cont.X,
            bins=4,
            ordered=False,
            labels=["x1", "x2", "x3", "x4"],
        )

        self.df_indep_cat_cat = self.df_indep_cont_cont.copy()
        self.df_indep_cat_cat.X = pd.cut(
            self.df_indep_cat_cat.X,
            bins=4,
            ordered=False,
            labels=["x1", "x2", "x3", "x4"],
        )
        self.df_indep_cat_cat.Y = pd.cut(
            self.df_indep_cat_cat.Y,
            bins=4,
            ordered=False,
            labels=["y1", "y2", "y3", "y4"],
        )

        self.df_indep_ord_cont = self.df_indep_cont_cont.copy()
        self.df_indep_ord_cont.X = pd.cut(self.df_indep_ord_cont.X, bins=4)

        self.model_dep = LinearGaussianBayesianNetwork(
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
        self.cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
        self.model_dep.add_cpds(self.cpd_z1, self.cpd_z2, self.cpd_z3, self.cpd_x, self.cpd_y_dep)
        self.df_dep = self.model_dep.simulate(n_samples=1000, seed=42)

        self.df_dep_cont_cont = self.df_dep.copy()
        self.df_dep_cont_cont.Z2 = pd.cut(
            self.df_dep_cont_cont.Z2,
            bins=4,
            ordered=False,
            labels=["z21", "z22", "z23", "z24"],
        )

        self.df_dep_cat_cont = self.df_dep_cont_cont.copy()
        self.df_dep_cat_cont.X = pd.cut(
            self.df_dep_cat_cont.X,
            bins=4,
            ordered=False,
            labels=["x1", "x2", "x3", "x4"],
        )

        self.df_dep_cat_cat = self.df_dep_cont_cont.copy()
        self.df_dep_cat_cat.X = pd.cut(
            self.df_dep_cat_cat.X,
            bins=4,
            ordered=False,
            labels=["x1", "x2", "x3", "x4"],
        )
        self.df_dep_cat_cat.Y = pd.cut(
            self.df_dep_cat_cat.Y,
            bins=4,
            ordered=False,
            labels=["y1", "y2", "y3", "y4"],
        )

        self.df_dep_ord_cont = self.df_dep_cont_cont.copy()
        self.df_dep_ord_cont.X = pd.cut(self.df_dep_ord_cont.X, bins=4)

    def test_pearsonr(self):
        coef, p_value = pearsonr(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=self.df_indep,
            boolean=False,
            seed=42,
        )
        self.assertTrue(abs(coef) <= 0.1)
        self.assertTrue(p_value >= 0.04)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=self.df_dep, boolean=False, seed=42)
        self.assertTrue(coef >= 0.1)
        self.assertTrue(np.isclose(p_value, 0, atol=1e-1))

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_no_cond(self):
        dep_coefs = [0.2038, 0.2038, 0.1733, 0.1527, 0.1733]
        dep_pvalues = [0, 0, 0, 0, 0]

        computed_coefs = []
        computed_pvalues = []
        for i, df_indep in enumerate(
            [
                self.df_indep,
                self.df_indep_cont_cont,
                self.df_indep_cat_cont,
                self.df_indep_cat_cat,
                self.df_indep_ord_cont,
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

        self.assertTrue(
            np.allclose(computed_coefs, dep_coefs, rtol=1e-2, atol=1e-2),
            msg=f"Non-conditional coefs mismatch at index {i}: {computed_coefs} != {dep_coefs}",
        )
        self.assertTrue(
            np.allclose(computed_pvalues, dep_pvalues, rtol=1e-2, atol=1e-2),
            msg=f"Non-conditional p-values mismatch at index {i}: {computed_pvalues} != {dep_pvalues}",
        )

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_indep(self):
        indep_coefs = [0.0014, 0.0023, 0.0041, 0.0213, 0.0041]
        indep_pvalues = [0.2430, 0.0161, 0.0522, 0.0184, 0.0522]

        computed_coefs = []
        computed_pvalues = []
        for i, df_indep in enumerate(
            [
                self.df_indep,
                self.df_indep_cont_cont,
                self.df_indep_cat_cont,
                self.df_indep_cat_cat,
                self.df_indep_ord_cont,
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

        self.assertTrue(
            np.allclose(computed_coefs, indep_coefs, rtol=1e-2, atol=1e-2),
            msg=f"Conditional (indep) coefs mismatch at index {i}: {computed_coefs} != {indep_coefs}",
        )
        self.assertTrue(
            np.allclose(computed_pvalues, indep_pvalues, rtol=1e-2, atol=1e-2),
            msg=f"Conditional (indep) p-values mismatch at index {i}: {computed_pvalues} != {indep_pvalues}",
        )

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_dependent(self):
        dep_coefs = np.array([0.1322, 0.1609, 0.1182, 0.1330, 0.1182])
        dep_pvalues = np.array([0, 0, 0, 0, 0])

        computed_coefs = []
        computed_pvalues = []
        for i, df_dep in enumerate(
            [
                self.df_dep,
                self.df_dep_cont_cont,
                self.df_dep_cat_cont,
                self.df_dep_cat_cat,
                self.df_dep_ord_cont,
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

        self.assertTrue(
            np.allclose(computed_coefs, dep_coefs, rtol=1e-2, atol=1e-2),
            msg=f"Conditional (dep) coefs mismatch at index {i}: {computed_coefs} != {dep_coefs}",
        )
        self.assertTrue(
            np.allclose(computed_pvalues, dep_pvalues, rtol=1e-2, atol=1e-2),
            msg=f"Conditional (dep) p-values mismatch at index {i}: {computed_pvalues} != {dep_pvalues}",
        )

    def test_gcm(self):
        # Non-conditional tests
        coef, p_value = gcm(
            X="X",
            Y="Y",
            Z=[],
            data=self.df_indep,
            boolean=False,
            seed=42,
        )
        self.assertAlmostEqual(round(coef, 3), 13.693)
        self.assertAlmostEqual(p_value, 0.0)

        # Conditional tests
        coef, p_value = gcm(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=self.df_indep,
            boolean=False,
            seed=42,
        )

        self.assertAlmostEqual(round(coef, 3), 0.097)
        self.assertEqual(round(p_value, 4), 0.9228)

        # Conditional tests
        coef, p_value = gcm(X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=self.df_dep, boolean=False, seed=42)

        self.assertAlmostEqual(round(coef, 3), 11.69)
        self.assertAlmostEqual(p_value, 0.0)

    def test_pearsonr_equivalence(self):
        is_independent = pearsonr_equivalence(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=self.df_dep,
            boolean=True,
            significance_level=0.05,
            delta_th=0.3,
        )
        self.assertFalse(is_independent)

        is_independent = pearsonr_equivalence(
            X="X",
            Y="Y",
            Z=["Z1", "Z2", "Z3"],
            data=self.df_dep,
            boolean=False,
            significance_level=0.05,
            delta_th=0.5,
        )
        self.assertTrue(is_independent)


class TestRegressionBasedLR(unittest.TestCase):
    """Tests for regression_based_lr conditional independence test."""

    @classmethod
    def setUpClass(cls):
        """Create reusable synthetic datasets."""
        rng = np.random.default_rng(2024)
        n = 1000

        # --- Continuous data: X \u2190 Z \u2192 Y  (so X \u27c2 Y | Z) --------
        Z_cont = rng.standard_normal(n)
        X_cont = 2.0 * Z_cont + rng.standard_normal(n) * 0.5
        Y_cont = 3.0 * Z_cont + rng.standard_normal(n) * 0.5
        cls.cont_data = pd.DataFrame({"X": X_cont, "Y": Y_cont, "Z": Z_cont})

        # --- Continuous data: direct dependence X \u2192 Y  -----------
        X_dep = rng.standard_normal(n)
        Y_dep = 2.0 * X_dep + rng.standard_normal(n) * 0.3
        cls.cont_dep_data = pd.DataFrame({"X": X_dep, "Y": Y_dep})

        # --- Binary X ------------------------------------------
        Z_bin = rng.standard_normal(n)
        prob = 1.0 / (1.0 + np.exp(-(2.0 * Z_bin)))
        X_bin = rng.binomial(1, prob, size=n)
        Y_bin = 3.0 * Z_bin + rng.standard_normal(n)
        cls.bin_data = pd.DataFrame(
            {
                "X": pd.Categorical(X_bin),
                "Y": Y_bin,
                "Z": Z_bin,
            }
        )

        # --- Multinomial X (3 classes) ----------------------------
        Z_multi = rng.standard_normal(n)
        logits = np.column_stack([np.zeros(n), 1.5 * Z_multi, -1.0 * Z_multi])
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        X_multi = np.array([rng.choice(3, p=probs[i]) for i in range(n)])
        Y_multi = 2.0 * Z_multi + rng.standard_normal(n)
        cls.multi_data = pd.DataFrame(
            {
                "X": pd.Categorical(X_multi),
                "Y": Y_multi,
                "Z": Z_multi,
            }
        )

        # --- Mixed: categorical Y, continuous X --------------------
        X_mixed = rng.standard_normal(n)
        Y_cat = rng.choice(["A", "B", "C"], size=n)
        cls.mixed_data = pd.DataFrame({"X": X_mixed, "Y": Y_cat})

    # ====== CONTINUOUS X TESTS =======================================

    def test_continuous_conditional_independence(self):
        """X \u27c2 Y | Z should be detected (high p-value)."""
        result = regression_based_lr(
            "X",
            "Y",
            ["Z"],
            self.cont_data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertTrue(result)

    def test_continuous_conditional_independence_tuple(self):
        """Return (stat, p, dof) format."""
        stat, p, dof = regression_based_lr(
            "X",
            "Y",
            ["Z"],
            self.cont_data,
            boolean=False,
        )
        self.assertIsInstance(stat, float)
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.05)
        self.assertEqual(dof, 1)

    def test_continuous_marginal_dependence(self):
        """X and Y are marginally dependent (no conditioning)."""
        result = regression_based_lr(
            "X",
            "Y",
            [],
            self.cont_data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertFalse(result)

    def test_continuous_direct_dependence(self):
        """X \u2192 Y, no conditioning, should be dependent."""
        stat, p, dof = regression_based_lr(
            "X",
            "Y",
            [],
            self.cont_dep_data,
            boolean=False,
        )
        self.assertLess(p, 0.01)

    # ====== BINARY X TESTS ===========================================

    def test_binary_conditional_independence(self):
        """Binary X \u27c2 Y | Z."""
        result = regression_based_lr(
            "X",
            "Y",
            ["Z"],
            self.bin_data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertTrue(result)

    def test_binary_marginal_dependence(self):
        """Binary X \u27c2\u0337 Y marginally (because Z is confounder)."""
        result = regression_based_lr(
            "X",
            "Y",
            [],
            self.bin_data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertFalse(result)

    # ====== MULTINOMIAL X TESTS =======================================

    def test_multinomial_conditional_independence(self):
        """Multinomial X \u27c2 Y | Z."""
        result = regression_based_lr(
            "X",
            "Y",
            ["Z"],
            self.multi_data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertTrue(result)

    def test_multinomial_returns_tuple(self):
        """Check tuple output with multinomial X."""
        stat, p, dof = regression_based_lr(
            "X",
            "Y",
            ["Z"],
            self.multi_data,
            boolean=False,
        )
        self.assertIsInstance(stat, float)
        self.assertGreaterEqual(stat, 0.0)
        # dof = q_y * (K-1) = 1 * 2 = 2
        self.assertEqual(dof, 2)

    # ====== MIXED DATA TESTS =========================================

    def test_mixed_categorical_Y(self):
        """Continuous X with categorical Y (independent)."""
        result = regression_based_lr(
            "X",
            "Y",
            [],
            self.mixed_data,
            boolean=True,
            significance_level=0.05,
        )
        # X and Y are independent by construction
        self.assertTrue(result)

    # ====== EDGE CASES ================================================

    def test_empty_conditioning_set_none(self):
        """Z=None should behave like Z=[]."""
        stat1, p1, _ = regression_based_lr(
            "X",
            "Y",
            None,
            self.cont_dep_data,
            boolean=False,
        )
        stat2, p2, _ = regression_based_lr(
            "X",
            "Y",
            [],
            self.cont_dep_data,
            boolean=False,
        )
        np.testing.assert_almost_equal(stat1, stat2, decimal=10)
        np.testing.assert_almost_equal(p1, p2, decimal=10)

    def test_string_Z(self):
        """Z passed as a single string instead of list."""
        result = regression_based_lr(
            "X",
            "Y",
            "Z",
            self.cont_data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertTrue(result)

    def test_multiple_conditioning_variables(self):
        """Conditioning on more than one variable."""
        rng = np.random.default_rng(99)
        n = 800
        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        X = Z1 + Z2 + rng.standard_normal(n) * 0.5
        Y = Z1 - Z2 + rng.standard_normal(n) * 0.5
        data = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

        result = regression_based_lr(
            "X",
            "Y",
            ["Z1", "Z2"],
            data,
            boolean=True,
            significance_level=0.05,
        )
        self.assertTrue(result)

    def test_missing_data_handled(self):
        """Rows with NaN should be dropped gracefully."""
        data = self.cont_data.copy()
        data.loc[0, "X"] = np.nan
        data.loc[5, "Z"] = np.nan
        stat, p, dof = regression_based_lr(
            "X",
            "Y",
            ["Z"],
            data,
            boolean=False,
        )
        self.assertIsInstance(p, float)
        self.assertFalse(np.isnan(p))

    def test_significance_level_respected(self):
        """Different significance levels give different boolean answers."""
        stat, p, _ = regression_based_lr(
            "X",
            "Y",
            [],
            self.cont_data,
            boolean=False,
        )
        # Use p-value itself as threshold boundary
        res_strict = regression_based_lr(
            "X",
            "Y",
            [],
            self.cont_data,
            boolean=True,
            significance_level=p + 1e-10,
        )
        res_lax = regression_based_lr(
            "X",
            "Y",
            [],
            self.cont_data,
            boolean=True,
            significance_level=p - 1e-10,
        )
        self.assertFalse(res_strict)
        self.assertTrue(res_lax)

    def test_asymmetry_acknowledged(self):
        """The test is asymmetric by design — document and verify the behavior."""
        stat_xy, p_xy, _ = regression_based_lr("X", "Y", ["Z"], self.cont_data, boolean=False)
        stat_yx, p_yx, _ = regression_based_lr("Y", "X", ["Z"], self.cont_data, boolean=False)
        # Both should agree on the independence verdict even if statistics differ
        self.assertGreaterEqual(p_xy, 0.05)
        self.assertGreaterEqual(p_yx, 0.05)

    def test_constant_variable(self):
        """Constant X should not crash — returns independence (no variation)."""
        rng = np.random.default_rng(7)
        data = pd.DataFrame({"X": [1.0] * 200, "Y": rng.standard_normal(200)})
        result = regression_based_lr("X", "Y", [], data, boolean=True, significance_level=0.05)
        self.assertTrue(result)

    def test_boolean_column_treated_as_categorical(self):
        """Boolean dtype columns should be treated as categorical (binary logistic)."""
        rng = np.random.default_rng(42)
        n = 500
        Z = rng.standard_normal(n)
        X = Z > 0  # boolean Series
        Y = 2.0 * Z + rng.standard_normal(n) * 0.3
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        result = regression_based_lr("X", "Y", ["Z"], data, boolean=True, significance_level=0.05)
        self.assertTrue(result)

    def test_invalid_column_name_raises(self):
        """Non-existent column in X, Y, or Z should raise ValueError."""
        with self.assertRaises(ValueError, msg="Unknown X should raise"):
            regression_based_lr("NONEXISTENT", "Y", [], self.cont_data)
        with self.assertRaises(ValueError, msg="Unknown Y should raise"):
            regression_based_lr("X", "NONEXISTENT", [], self.cont_data)
        with self.assertRaises(ValueError, msg="Unknown Z should raise"):
            regression_based_lr("X", "Y", ["NONEXISTENT"], self.cont_data)

    def test_x_equals_y_raises(self):
        """Passing the same variable for both X and Y should raise ValueError."""
        with self.assertRaises(ValueError):
            regression_based_lr("X", "X", [], self.cont_data)

    def test_lookup_by_string_name(self):
        """regression_based_lr should be retrievable from ci_registry by string name."""
        fn = ci_registry.get_test("regression_based_lr")
        self.assertIs(fn, regression_based_lr)
        self.assertIn("regression_based_lr", ci_registry.list_all())
        self.assertIn("regression_based_lr", ci_registry.list_all(data_type="mixed"))
        self.assertIn("regression_based_lr", ci_registry.list_all(data_type="continuous"))
        self.assertIn("regression_based_lr", ci_registry.list_all(data_type="discrete"))


class TestRegressionBasedLRIntegration(unittest.TestCase):
    """Integration tests: use regression_based_lr with PC algorithm."""

    def test_pc_with_regression_lr_continuous(self):
        """PC algorithm discovers correct skeleton on continuous data."""
        from pgmpy.estimators import PC

        rng = np.random.default_rng(123)
        n = 2000
        Z = rng.standard_normal(n)
        X = 2 * Z + rng.standard_normal(n) * 0.3
        Y = 3 * Z + rng.standard_normal(n) * 0.3
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        est = PC(data)
        model, _ = est.estimate(
            ci_test=regression_based_lr,  # pass the callable
            significance_level=0.05,
            return_type="skeleton",
        )
        edges = set(model.edges())
        # Expected skeleton: X\u2014Z, Y\u2014Z  (no X\u2014Y edge)
        self.assertFalse(
            ("X", "Y") in edges or ("Y", "X") in edges,
            "Spurious X\u2014Y edge should not exist.",
        )
        self.assertTrue(
            ("X", "Z") in edges or ("Z", "X") in edges,
            "X\u2014Z edge should exist.",
        )
