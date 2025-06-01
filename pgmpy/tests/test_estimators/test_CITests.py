import math
import os
import unittest

import numpy as np
import pandas as pd
import pytest
from numpy import testing as np_test

from pgmpy.estimators.CITests import *
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

np.random.seed(42)
ON_GITHUB_RUNNER = os.getenv("GITHUB_ACTIONS") == "true"


class TestPearsonr(unittest.TestCase):
    def setUp(self):
        self.df_ind = pd.DataFrame(np.random.randn(10000, 3), columns=["X", "Y", "Z"])

        Z = np.random.randn(10000)
        X = 3 * Z + np.random.normal(loc=0, scale=0.1, size=10000)
        Y = 2 * Z + np.random.normal(loc=0, scale=0.1, size=10000)

        self.df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        Z1 = np.random.randn(10000)
        Z2 = np.random.randn(10000)
        X = 3 * Z1 + 2 * Z2 + np.random.normal(loc=0, scale=0.1, size=10000)
        Y = 2 * Z1 + 3 * Z2 + np.random.normal(loc=0, scale=0.1, size=10000)
        self.df_cind_mul = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

        X = np.random.rand(10000)
        Y = np.random.rand(10000)
        Z = 2 * X + 2 * Y + np.random.normal(loc=0, scale=0.1, size=10000)
        self.df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_pearsonr(self):
        coef, p_value = pearsonr(X="X", Y="Y", Z=[], data=self.df_ind, boolean=False)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z"], data=self.df_cind, boolean=False
        )
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z1", "Z2"], data=self.df_cind_mul, boolean=False
        )
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z"], data=self.df_vstruct, boolean=False
        )
        self.assertTrue(abs(coef) > 0.9)
        self.assertTrue(p_value < 0.05)

        # Tests for when boolean=True
        self.assertTrue(
            pearsonr(X="X", Y="Y", Z=[], data=self.df_ind, significance_level=0.05)
        )
        self.assertTrue(
            pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_cind, significance_level=0.05)
        )
        self.assertTrue(
            pearsonr(
                X="X",
                Y="Y",
                Z=["Z1", "Z2"],
                data=self.df_cind_mul,
                significance_level=0.05,
            )
        )
        self.assertFalse(
            pearsonr(
                X="X", Y="Y", Z=["Z"], data=self.df_vstruct, significance_level=0.05
            )
        )


class TestKernelCITests(unittest.TestCase):
    def setUp(self):
        # Create various test datasets
        n_samples = 3000  # Increase sample size for better statistics
        
        # 1. Linear independence dataset
        self.df_lin_ind = pd.DataFrame(np.random.randn(n_samples, 3), columns=["X", "Y", "Z"])
        
        # 2. Linear conditional independence - ensure proper CI structure
        np.random.seed(42)
        Z = np.random.randn(n_samples)
        # Add more noise to ensure conditional independence
        noise_x = np.random.randn(n_samples)
        noise_y = np.random.randn(n_samples)
        X = 3 * Z + 2 * noise_x  # Increase noise coefficient
        Y = 2 * Z + 2 * noise_y  # Increase noise coefficient
        self.df_lin_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        
        # 3. Non-linear dependence (quadratic)
        X = np.random.uniform(-2, 2, n_samples)
        Y = X**2 + np.random.normal(loc=0, scale=0.5, size=n_samples)
        self.df_nonlin_dep = pd.DataFrame({"X": X, "Y": Y})
        
        # 4. Non-linear conditional independence - more careful construction
        Z = np.random.uniform(-3, 3, n_samples)
        noise_x = np.random.randn(n_samples) * 0.5
        noise_y = np.random.randn(n_samples) * 0.5
        X = np.sin(2 * Z) + noise_x
        Y = np.cos(2 * Z) + noise_y
        self.df_nonlin_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        
        # 5. Non-linear conditional dependence
        Z = np.random.randn(n_samples)
        X = np.sin(Z) + np.random.normal(loc=0, scale=0.3, size=n_samples)
        # Make Y depend on both X and Z
        Y = 0.5 * X + np.cos(Z) + np.random.normal(loc=0, scale=0.3, size=n_samples)
        self.df_nonlin_cdep = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        
        # 6. Multiple conditioning variables
        Z1 = np.random.randn(n_samples)
        Z2 = np.random.randn(n_samples)
        noise_x = np.random.randn(n_samples) * 0.5
        noise_y = np.random.randn(n_samples) * 0.5
        X = np.sin(Z1) + np.exp(-Z2**2/2) + noise_x
        Y = np.cos(Z1) + np.tanh(Z2) + noise_y
        self.df_multi_nonlin_cind = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})
        
        # 7. V-structure
        X = np.random.randn(n_samples)
        Y = np.random.randn(n_samples)
        Z = np.sin(X) + np.cos(Y) + np.random.normal(loc=0, scale=0.3, size=n_samples)
        self.df_nonlin_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_rcit_unconditional(self):
        """Test RCIT for unconditional independence/dependence."""
        # Test linear independence
        stat, p_value = rcit(X="X", Y="Y", Z=[], data=self.df_lin_ind, boolean=False, seed=42)
        self.assertTrue(p_value > 0.05, f"Expected p_value > 0.05, got {p_value}")
        
        # Test non-linear dependence
        stat, p_value = rcit(X="X", Y="Y", Z=[], data=self.df_nonlin_dep, boolean=False, seed=42)
        self.assertTrue(p_value < 0.05, f"Expected p_value < 0.05, got {p_value}")
        
        # Boolean tests
        self.assertTrue(
            rcit(X="X", Y="Y", Z=[], data=self.df_lin_ind, significance_level=0.05, seed=42)
        )
        self.assertFalse(
            rcit(X="X", Y="Y", Z=[], data=self.df_nonlin_dep, significance_level=0.05, seed=42)
        )

    def test_rcit_conditional(self):
        """Test RCIT for conditional independence/dependence."""
        # Test linear conditional independence
        stat, p_value = rcit(X="X", Y="Y", Z=["Z"], data=self.df_lin_cind, boolean=False, seed=42)
        self.assertTrue(p_value > 0.01, f"Linear CI: Expected p_value > 0.01, got {p_value}")
        
        # Test non-linear conditional independence
        stat, p_value = rcit(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=42)
        self.assertTrue(p_value > 0.01, f"Non-linear CI: Expected p_value > 0.01, got {p_value}")
        
        # Test non-linear conditional dependence
        stat, p_value = rcit(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cdep, boolean=False, seed=42)
        self.assertTrue(p_value < 0.01, f"Non-linear CD: Expected p_value < 0.01, got {p_value}")
        

    def test_rcit_multiple_conditioning(self):
        """Test RCIT with multiple conditioning variables."""
        stat, p_value = rcit(
            X="X", Y="Y", Z=["Z1", "Z2"], 
            data=self.df_multi_nonlin_cind, 
            boolean=False, 
            seed=42
        )
        self.assertTrue(p_value > 0.05, f"Multi-var CI: Expected p_value > 0.05, got {p_value}")

    def test_rcot_unconditional(self):
        """Test RCoT for unconditional independence/dependence."""
        # Test linear independence
        stat, p_value = rcot(X="X", Y="Y", Z=[], data=self.df_lin_ind, boolean=False, seed=42)
        self.assertTrue(p_value > 0.05, f"Expected p_value > 0.05, got {p_value}")
        
        # Test non-linear dependence
        stat, p_value = rcot(X="X", Y="Y", Z=[], data=self.df_nonlin_dep, boolean=False, seed=42)
        self.assertTrue(p_value < 0.05, f"Expected p_value < 0.05, got {p_value}")

    def test_rcot_conditional(self):
        """Test RCoT for conditional independence/dependence."""
        # Test linear conditional independence
        stat, p_value = rcot(X="X", Y="Y", Z=["Z"], data=self.df_lin_cind, boolean=False, seed=42)
        self.assertTrue(p_value > 0.01, f"Linear CI: Expected p_value > 0.01, got {p_value}")
        
        # Test non-linear conditional independence
        stat, p_value = rcot(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=42)
        self.assertTrue(p_value > 0.01, f"Non-linear CI: Expected p_value > 0.01, got {p_value}")
        
        # Test non-linear conditional dependence
        stat, p_value = rcot(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cdep, boolean=False, seed=42)
        self.assertTrue(p_value < 0.05, f"Non-linear CD: Expected p_value < 0.05, got {p_value}")

    def test_kernel_tests_vs_pearsonr(self):
        """Compare kernel tests with Pearson on linear relationships."""
        # For linear relationships, all tests should give similar results
        tests = [
            (pearsonr, "pearsonr"),
            (rcit, "rcit"),
            (rcot, "rcot")
        ]
        
        for test_func, test_name in tests:
            # Linear independence
            if test_name == "pearsonr":
                _, p_value = test_func(X="X", Y="Y", Z=[], data=self.df_lin_ind, boolean=False)
            else:
                _, p_value = test_func(X="X", Y="Y", Z=[], data=self.df_lin_ind, boolean=False, seed=42)
            self.assertTrue(p_value > 0.05, f"{test_name} failed on linear independence")
            
            # Linear conditional independence
            if test_name == "pearsonr":
                _, p_value = test_func(X="X", Y="Y", Z=["Z"], data=self.df_lin_cind, boolean=False)
            else:
                _, p_value = test_func(X="X", Y="Y", Z=["Z"], data=self.df_lin_cind, boolean=False, seed=42)
            self.assertTrue(p_value > 0.05, f"{test_name} failed on linear conditional independence")

    def test_kernel_tests_hyperparameters(self):
        """Test kernel tests with different hyperparameters."""
        results = []
        # Test with different number of random features
        for num_f in [50, 100, 200]:
            stat, p_value = rcit(
                X="X", Y="Y", Z=["Z"], 
                data=self.df_nonlin_cind, 
                boolean=False,
                num_f=num_f,
                num_f2=5,
                seed=42
            )
            results.append((num_f, p_value))
        
        # Test with different approximation methods
        for approx in ["lpd4", "gamma", "hbe"]:
            stat, p_value = rcit(
                X="X", Y="Y", Z=["Z"], 
                data=self.df_nonlin_cind, 
                boolean=False,
                approx=approx,
                seed=42
            )
            self.assertTrue(p_value > 0.05, f"Failed with approx={approx}: p_value={p_value}")

        # Check that at least one configuration shows independence
        # Different num_f values might give different results due to approximation
        max_p_value = max(p for _, p in results)
        self.assertTrue(max_p_value > 0.01, 
            f"All configurations failed to detect independence: {results}")

    def test_kernel_tests_edge_cases(self):
        """Test kernel tests with edge cases."""
        # Test with constant variable
        df_const = self.df_lin_ind.copy()
        df_const["C"] = 1.0
        
        # Should return independence (p_value = 1) when one variable is constant
        stat, p_value = rcit(X="X", Y="C", Z=[], data=df_const, boolean=False, seed=42)
        self.assertEqual(p_value, 1.0, "Expected p_value=1 for constant variable")
        
        # Test with very small sample size
        df_small = self.df_nonlin_dep.iloc[:10]
        stat, p_value = rcit(X="X", Y="Y", Z=[], data=df_small, boolean=False, seed=42)
        self.assertIsInstance(p_value, float)
        self.assertTrue(0 <= p_value <= 1)
        
        # Test with single sample conditioning variable
        df_single = pd.DataFrame({
            "X": np.random.randn(100),
            "Y": np.random.randn(100),
            "Z": np.random.randn(100)
        })
        stat, p_value = rcit(X="X", Y="Y", Z=["Z"], data=df_single, boolean=False, seed=42)
        self.assertIsInstance(p_value, float)
        self.assertTrue(0 <= p_value <= 1)

    def test_kernel_tests_reproducibility(self):
        """Test that results are reproducible with same seed."""
        # RCIT
        stat1, p1 = rcit(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=42)
        stat2, p2 = rcit(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=42)
        self.assertEqual(stat1, stat2, "RCIT statistics should be equal with same seed")
        self.assertEqual(p1, p2, "RCIT p-values should be equal with same seed")
        
        # RCoT
        stat1, p1 = rcot(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=42)
        stat2, p2 = rcot(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=42)
        self.assertEqual(stat1, stat2, "RCoT statistics should be equal with same seed")
        self.assertEqual(p1, p2, "RCoT p-values should be equal with same seed")
        
        # Different seeds should give different results
        stat3, p3 = rcit(X="X", Y="Y", Z=["Z"], data=self.df_nonlin_cind, boolean=False, seed=123)
        self.assertNotEqual(stat1, stat3, "RCIT statistics should differ with different seeds")


class TestDiscreteTests(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    def test_chisquare_adult_dataset(self):
        # Comparison values taken from dagitty (DAGitty)
        coef, p_value, dof = chi_square(
            X="Age", Y="Immigrant", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 57.75, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -25.47, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(
            X="Age", Y="Race", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 56.25, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -24.75, decimal=1)
        self.assertEqual(dof, 4)

        coef, p_value, dof = chi_square(
            X="Age", Y="Sex", Z=[], data=self.df_adult, boolean=False
        )
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

        coef, p_value, dof = chi_square(
            X="Immigrant", Y="Sex", Z=[], data=self.df_adult, boolean=False
        )
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


class TestResidualMethod(unittest.TestCase):
    def setUp(self):
        # Create a combination of mixed data types

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
        self.cpd_y_indep = LinearGaussianCPD(
            "Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"]
        )
        self.model_indep.add_cpds(
            self.cpd_z1, self.cpd_z2, self.cpd_z3, self.cpd_x, self.cpd_y_indep
        )
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
        self.cpd_y_dep = LinearGaussianCPD(
            "Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"]
        )
        self.model_dep.add_cpds(
            self.cpd_z1, self.cpd_z2, self.cpd_z3, self.cpd_x, self.cpd_y_dep
        )
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

        coef, p_value = pearsonr(
            X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=self.df_dep, boolean=False, seed=42
        )
        self.assertTrue(coef >= 0.1)
        self.assertTrue(np.isclose(p_value, 0, atol=1e-1))

    def test_pillai(self):
        # Non-conditional tests
        dep_coefs = [0.1572, 0.1572, 0.1523, 0.1468, 0.1523]
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

        # Conditional tests (independent case)
        indep_coefs = [0.0014, 0.0023, 0.0041, 0.0213, 0.0041]
        indep_pvalues = [0.3086, 0.1277, 0.2498, 0.0114, 0.2498]

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

        # Conditional tests (dependent case)
        dep_coefs = [0.1322, 0.1609, 0.1158, 0.1188, 0.1158]
        dep_pvalues = [0, 0, 0, 0, 0]

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
        self.assertAlmostEqual(round(coef, 3), 11.934)
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

        self.assertAlmostEqual(round(coef, 3), -1.908)
        self.assertEqual(round(p_value, 4), 0.0564)

        # Conditional tests
        coef, p_value = gcm(
            X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=self.df_dep, boolean=False, seed=42
        )

        self.assertAlmostEqual(round(coef, 3), 11.69)
        self.assertAlmostEqual(p_value, 0.0)


class TestGetCITest(unittest.TestCase):
    """Test get_ci_test function to ensure all tests are properly registered."""
    
    def test_supported_tests(self):
        """Test that all supported tests can be retrieved."""
        supported_tests = [
            "chi_square",
            "g_sq", 
            "log_likelihood",
            "modified_log_likelihood",
            "pearsonr",
            "pillai",
            "gcm",
            "rcit",
            "rcot",
        ]
        
        for test_name in supported_tests:
            test_func = get_ci_test(test_name)
            self.assertIsNotNone(test_func, f"Failed to get test function for {test_name}")
            self.assertTrue(callable(test_func), f"{test_name} is not callable")
    
    def test_unsupported_test(self):
        """Test that unsupported test names raise ValueError."""
        with self.assertRaises(ValueError):
            get_ci_test("unsupported_test")
    
    def test_callable_test(self):
        """Test that callable functions are returned as-is."""
        def custom_test(X, Y, Z, data, **kwargs):
            return True
        
        result = get_ci_test(custom_test)
        self.assertEqual(result, custom_test)
    
    def test_case_insensitive(self):
        """Test that test names are case-insensitive."""
        test_func_lower = get_ci_test("rcit")
        test_func_upper = get_ci_test("RCIT")
        test_func_mixed = get_ci_test("RcIt")
        
        self.assertEqual(test_func_lower, test_func_upper)
        self.assertEqual(test_func_lower, test_func_mixed)


if __name__ == "__main__":
    unittest.main()