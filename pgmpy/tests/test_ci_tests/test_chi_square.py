import unittest

import numpy as np
import pandas as pd
from numpy import testing as np_test

from pgmpy.ci_tests import ChiSquare


class TestChiSquare(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
        self.test = ChiSquare(data=self.df_adult)

    def test_chisquare_adult_dataset(self):
        # Comparison values taken from dagitty (DAGitty)
        self.test("Age", "Immigrant", [])
        np_test.assert_almost_equal(self.test.statistic_, 57.75, decimal=1)
        np_test.assert_almost_equal(np.log(self.test.p_value_), -25.47, decimal=1)
        self.assertEqual(self.test.dof_, 4)

        self.test("Age", "Race", [])
        np_test.assert_almost_equal(self.test.statistic_, 56.25, decimal=1)
        np_test.assert_almost_equal(np.log(self.test.p_value_), -24.75, decimal=1)
        self.assertEqual(self.test.dof_, 4)

        self.test("Age", "Sex", [])
        np_test.assert_almost_equal(self.test.statistic_, 289.62, decimal=1)
        np_test.assert_almost_equal(np.log(self.test.p_value_), -139.82, decimal=1)
        self.assertEqual(self.test.dof_, 4)

        self.test(
            "Education",
            "HoursPerWeek",
            ["Age", "Immigrant", "Race", "Sex"],
        )
        np_test.assert_almost_equal(self.test.statistic_, 1460.11, decimal=1)
        np_test.assert_almost_equal(self.test.p_value_, 0, decimal=1)
        self.assertEqual(self.test.dof_, 316)

        self.test("Immigrant", "Sex", [])
        np_test.assert_almost_equal(self.test.statistic_, 0.2724, decimal=1)
        np_test.assert_almost_equal(np.log(self.test.p_value_), -0.50, decimal=1)
        self.assertEqual(self.test.dof_, 1)

        self.test("Education", "MaritalStatus", ["Age", "Sex"])
        np_test.assert_almost_equal(self.test.statistic_, 481.96, decimal=1)
        np_test.assert_almost_equal(self.test.p_value_, 0, decimal=1)
        self.assertEqual(self.test.dof_, 58)

        # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
        # dataframes with very few samples. Update: Might be same from scipy=1.7.0
        self.test(
            "Income",
            "Race",
            ["Age", "Education", "HoursPerWeek", "MaritalStatus"],
        )
        np_test.assert_almost_equal(self.test.statistic_, 66.39, decimal=1)
        np_test.assert_almost_equal(self.test.p_value_, 0.99, decimal=1)
        self.assertEqual(self.test.dof_, 136)

        self.test(
            "Immigrant",
            "Income",
            ["Age", "Education", "HoursPerWeek", "MaritalStatus"],
        )
        np_test.assert_almost_equal(self.test.statistic_, 65.59, decimal=1)
        np_test.assert_almost_equal(self.test.p_value_, 0.999, decimal=2)
        self.assertEqual(self.test.dof_, 131)

    def test_discrete_tests(self):
        self.assertFalse(self.test("Age", "Immigrant", [], significance_level=0.05))
        self.assertFalse(self.test("Age", "Race", [], significance_level=0.05))
        self.assertFalse(self.test("Age", "Sex", [], significance_level=0.05))
        self.assertFalse(
            self.test(
                "Education",
                "HoursPerWeek",
                ["Age", "Immigrant", "Race", "Sex"],
                significance_level=0.05,
            )
        )
        self.assertTrue(self.test("Immigrant", "Sex", [], significance_level=0.05))
        self.assertFalse(self.test("Education", "MaritalStatus", ["Age", "Sex"], significance_level=0.05))

    def test_exactly_same_vars(self):
        x = np.random.choice([0, 1], size=1000)
        y = x.copy()
        df = pd.DataFrame({"x": x, "y": y})

        test = ChiSquare(data=df)
        test("x", "y", [])
        self.assertEqual(test.dof_, 1)
        np_test.assert_almost_equal(test.p_value_, 0, decimal=5)
