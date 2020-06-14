import unittest

import numpy as np
import pandas as pd
from numpy import testing as np_test

from pgmpy.estimators.CITests import pearsonr, chi_square


np.random.seed(42)


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
        coef, p_value = pearsonr(X="X", Y="Y", Z=[], data=self.df_ind)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_cind)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z1", "Z2"], data=self.df_cind_mul)
        self.assertTrue(coef < 0.1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_vstruct)
        self.assertTrue(abs(coef) > 0.9)
        self.assertTrue(p_value < 0.05)


class TestChiSquare(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    def test_chisquare_adult_dataset(self):
        # Comparision values taken from dagitty (DAGitty)
        coef, p_value = chi_square(X="Age", Y="Immigrant", Z=[], data=self.df_adult)
        np_test.assert_almost_equal(coef, 57.75, decimal=1)
        self.assertTrue(p_value < 0.05)

        coef, p_value = chi_square(X="Age", Y="Race", Z=[], data=self.df_adult)
        np_test.assert_almost_equal(coef, 56.25, decimal=1)
        self.assertTrue(p_value < 0.05)

        coef, p_value = chi_square(X="Age", Y="Sex", Z=[], data=self.df_adult)
        np_test.assert_almost_equal(coef, 289.62, decimal=1)
        self.assertTrue(p_value < 0.05)

        coef, p_value = chi_square(
            X="Education",
            Y="HoursPerWeek",
            Z=["Age", "Immigrant", "Race", "Sex"],
            data=self.df_adult,
        )
        np_test.assert_almost_equal(coef, 1460.11, decimal=1)
        self.assertTrue(p_value < 0.05)

        coef, p_value = chi_square(X="Immigrant", Y="Sex", Z=[], data=self.df_adult)
        np_test.assert_almost_equal(coef, 0.2724, decimal=1)
        self.assertTrue(p_value > 0.05)

        coef, p_value = chi_square(
            X="Education", Y="MaritalStatus", Z=["Age", "Sex"], data=self.df_adult
        )
        np_test.assert_almost_equal(coef, 481.96, decimal=1)
        self.assertTrue(p_value < 0.05)
