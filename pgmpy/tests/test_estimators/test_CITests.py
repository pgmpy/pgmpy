import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators.CITests import pearsonr


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

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_cind)
        self.assertTrue(coef < 0.1)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z1", "Z2"], data=self.df_cind_mul)
        self.assertTrue(coef < 0.1)

        coef, p_value = pearsonr(X="X", Y="Y", Z=["Z"], data=self.df_vstruct)
        self.assertTrue(abs(coef) > 0.9)
