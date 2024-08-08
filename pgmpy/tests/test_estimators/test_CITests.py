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
            freeman_tuckey,
            modified_log_likelihood,
            neyman,
            cressie_read,
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
            freeman_tuckey,
            modified_log_likelihood,
            neyman,
            cressie_read,
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
        self.df_indep = self.model_indep.simulate(n=1000, seed=42)

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
        self.df_dep = self.model_dep.simulate(n=1000, seed=42)

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

    @pytest.mark.skipif(ON_GITHUB_RUNNER, reason="Values differ on GitHub runner")
    def test_pillai(self):
        # Non-conditional tests
        dep_coefs = [0.1572, 0.1572, 0.1523, 0.1468, 0.1523]
        dep_pvalues = [0, 0, 0, 0.0, 0]
        for i, df_indep in enumerate(
            [
                self.df_indep,
                self.df_indep_cont_cont,
                self.df_indep_cat_cont,
                self.df_indep_cat_cat,
                self.df_indep_ord_cont,
            ]
        ):
            coef, p_value = ci_pillai(
                X="X",
                Y="Y",
                Z=[],
                data=df_indep,
                boolean=False,
                seed=42,
            )
            self.assertTrue(np.isclose(coef, dep_coefs[i], atol=1e-4))
            self.assertTrue(np.isclose(p_value, dep_pvalues[i], atol=1e-4))

        # Conditional tests
        indep_coefs = [0.0010, 0.0023, 0.0041, 0.0213, 0.0041]
        indep_pvalues = [0.3086, 0.1277, 0.2498, 0.0114, 0.2498]
        for i, df_indep in enumerate(
            [
                self.df_indep,
                self.df_indep_cont_cont,
                self.df_indep_cat_cont,
                self.df_indep_cat_cat,
                self.df_indep_ord_cont,
            ]
        ):
            coef, p_value = ci_pillai(
                X="X",
                Y="Y",
                Z=["Z1", "Z2", "Z3"],
                data=df_indep,
                boolean=False,
                seed=42,
            )
            self.assertTrue(np.isclose(coef, indep_coefs[i], atol=1e-4))
            self.assertTrue(np.isclose(p_value, indep_pvalues[i], atol=1e-4))

        dep_coefs = [0.1322, 0.1609, 0.1158, 0.1188, 0.1158]
        dep_pvalues = [0, 0, 0, 0, 0]
        for i, df_dep in enumerate(
            [
                self.df_dep,
                self.df_dep_cont_cont,
                self.df_dep_cat_cont,
                self.df_dep_cat_cat,
                self.df_dep_ord_cont,
            ]
        ):
            coef, p_value = ci_pillai(
                X="X", Y="Y", Z=["Z1", "Z2", "Z3"], data=df_dep, boolean=False, seed=42
            )
            self.assertTrue(np.isclose(coef, dep_coefs[i], atol=1e-4))
            self.assertTrue(np.isclose(p_value, dep_pvalues[i], atol=1e-4))
