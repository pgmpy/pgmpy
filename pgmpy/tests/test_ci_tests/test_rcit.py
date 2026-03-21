import os
import unittest

import numpy as np
import pandas as pd

from pgmpy.ci_tests import RCIT, RCoT


@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping RCIT tests on GitHub Actions.")
class TestRCIT(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 2000

        # X _|_ Y | Z  (Z is a common cause)
        z = rng.standard_normal(n)
        self.df_cind = pd.DataFrame({"X": z + rng.standard_normal(n), "Y": z + rng.standard_normal(n), "Z": z})

        # X -> Y, Z -> X, Z -> Y  (X and Y are dependent given Z)
        x_dep = z + rng.standard_normal(n)
        self.df_dep = pd.DataFrame({"X": x_dep, "Y": z + x_dep + rng.standard_normal(n), "Z": z})

    def test_rcit(self):
        test = RCIT(data=self.df_cind, seed=0)

        # Falls back to Pearsonr when Z is empty; X and Y are marginally dependent.
        test("X", "Y", [])
        self.assertLess(test.p_value_, 0.05)

        # Conditional independence: X _|_ Y | Z should give high p-value.
        test("X", "Y", ["Z"])
        self.assertGreater(test.p_value_, 0.05)

        # Conditional dependence: X and Y are NOT independent given Z.
        test = RCIT(data=self.df_dep, seed=0)
        test("X", "Y", ["Z"])
        self.assertLess(test.p_value_, 0.05)


@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping RCoT tests on GitHub Actions.")
class TestRCoT(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 2000

        # X _|_ Y | Z  (Z is a common cause)
        z = rng.standard_normal(n)
        self.df_cind = pd.DataFrame({"X": z + rng.standard_normal(n), "Y": z + rng.standard_normal(n), "Z": z})

        # X -> Y, Z -> X, Z -> Y  (X and Y are dependent given Z)
        x_dep = z + rng.standard_normal(n)
        self.df_dep = pd.DataFrame({"X": x_dep, "Y": z + x_dep + rng.standard_normal(n), "Z": z})

    def test_rcot(self):
        test = RCoT(data=self.df_cind, seed=0)

        # Falls back to Pearsonr when Z is empty; X and Y are marginally dependent.
        test("X", "Y", [])
        self.assertLess(test.p_value_, 0.05)

        # Conditional independence: X _|_ Y | Z should give high p-value.
        test("X", "Y", ["Z"])
        self.assertGreater(test.p_value_, 0.05)

        # Conditional dependence: X and Y are NOT independent given Z.
        test = RCoT(data=self.df_dep, seed=0)
        test("X", "Y", ["Z"])
        self.assertLess(test.p_value_, 0.05)
