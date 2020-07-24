import unittest
import math

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


class TestChiSquare(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    def test_chisquare_adult_dataset(self):
        # Comparision values taken from dagitty (DAGitty)
        coef, dof, p_value = chi_square(
            X="Age", Y="Immigrant", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 57.75, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -25.47, decimal=1)
        self.assertEqual(dof, 4)

        coef, dof, p_value = chi_square(
            X="Age", Y="Race", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 56.25, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -24.75, decimal=1)
        self.assertEqual(dof, 4)

        coef, dof, p_value = chi_square(
            X="Age", Y="Sex", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 289.62, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -139.82, decimal=1)
        self.assertEqual(dof, 4)

        coef, dof, p_value = chi_square(
            X="Education",
            Y="HoursPerWeek",
            Z=["Age", "Immigrant", "Race", "Sex"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 1460.11, decimal=1)
        # Really small value; hence rounded in this case.
        np_test.assert_almost_equal(math.ceil(np.log(p_value)), -335, decimal=1)
        self.assertEqual(dof, 316)

        coef, dof, p_value = chi_square(
            X="Immigrant", Y="Sex", Z=[], data=self.df_adult, boolean=False
        )
        np_test.assert_almost_equal(coef, 0.2724, decimal=1)
        np_test.assert_almost_equal(np.log(p_value), -0.50, decimal=1)
        self.assertEqual(dof, 1)

        coef, dof, p_value = chi_square(
            X="Education",
            Y="MaritalStatus",
            Z=["Age", "Sex"],
            data=self.df_adult,
            boolean=False,
        )
        np_test.assert_almost_equal(coef, 481.96, decimal=1)
        # Same here; Really small hence rounded
        np_test.assert_almost_equal(
            math.floor(np.log(p_value)), math.floor(-155.17), decimal=1
        )
        self.assertEqual(dof, 58)

        # Tests for when boolean=True
        self.assertFalse(
            chi_square(
                X="Age",
                Y="Immigrant",
                Z=[],
                data=self.df_adult,
                significance_level=0.05,
            )
        )

        self.assertFalse(
            chi_square(
                X="Age", Y="Race", Z=[], data=self.df_adult, significance_level=0.05
            )
        )

        self.assertFalse(
            chi_square(
                X="Age", Y="Sex", Z=[], data=self.df_adult, significance_level=0.05
            )
        )

        self.assertFalse(
            chi_square(
                X="Education",
                Y="HoursPerWeek",
                Z=["Age", "Immigrant", "Race", "Sex"],
                data=self.df_adult,
                significance_level=0.05,
            )
        )
        self.assertTrue(
            chi_square(
                X="Immigrant",
                Y="Sex",
                Z=[],
                data=self.df_adult,
                significance_level=0.05,
            )
        )
        self.assertFalse(
            chi_square(
                X="Education",
                Y="MaritalStatus",
                Z=["Age", "Sex"],
                data=self.df_adult,
                significance_level=0.05,
            )
        )

    def test_exactly_same_vars(self):
        x = np.random.choice([0, 1], size=1000)
        y = x.copy()
        df = pd.DataFrame({"x": x, "y": y})
        chi, dof, p_value = chi_square(X="x", Y="y", Z=[], data=df, boolean=False)
        np_test.assert_almost_equal(chi, 996.0, decimal=1)
        self.assertEqual(dof, 1)
        np_test.assert_almost_equal(p_value, 0, decimal=5)
