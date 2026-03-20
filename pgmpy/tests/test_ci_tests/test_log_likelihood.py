import unittest

import numpy as np
import pandas as pd
from numpy import testing as np_test

from pgmpy.ci_tests import LogLikelihood


class TestLogLikelihood(unittest.TestCase):
    def setUp(self):
        self.df_adult = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")
        self.test = LogLikelihood(data=self.df_adult)

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

        test = LogLikelihood(data=df)
        test("x", "y", [])
        self.assertEqual(test.dof_, 1)
        np_test.assert_almost_equal(test.p_value_, 0, decimal=5)
