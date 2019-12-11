import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators.CITests import ChiSquare, Pearsonr 


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
        pr1 = Pearsonr(data=self.df_ind)
        coef, p_value = pr1.compute_statistic(X="X", Y="Y", Z=[])
        self.assertTrue(coef < 0.1)

        pr2 = Pearsonr(data=self.df_cind)
        coef, p_value = pr2.compute_statistic(X="X", Y="Y", Z=["Z"])
        self.assertTrue(coef < 0.1)

        pr3 = Pearsonr(data=self.df_cind_mul)
        coef, p_value = pr3.compute_statistic(X="X", Y="Y", Z=["Z1", "Z2"])
        self.assertTrue(coef < 0.1)

        pr4 = Pearsonr(data=self.df_vstruct)
        coef, p_value = pr4.compute_statistic(X="X", Y="Y", Z=["Z"])
        self.assertTrue(abs(coef) > 0.9)

    def test_chi_square(self):
        data = pd.DataFrame(
            np.random.randint(0, 2, size=(1000, 4)), columns=list("ABCD")
        )
        data["E"] = data["A"] + data["B"] + data["C"]
        t = ChiSquare(data=data, tol=0.01)

        self.assertTrue(t.test_independence("A", "C"))  # independent
        self.assertTrue(t.test_independence("A", "B", "D"))  # independent
        self.assertFalse(
            t.test_independence("A", "B", ["D", "E"])
        )  # dependent

    def test_test_conditional_independence_titanic(self):
        titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        t = ChiSquare(data=titanic_data)

        self.assertTrue(t.test_independence("Embarked", "Sex"))
        self.assertFalse(
            t.test_independence("Pclass", "Survived", ["Embarked"])
        )
        self.assertTrue(
            t.test_independence("Embarked", "Survived", ["Sex", "Pclass"])
        )
        # insufficient data test commented out, because generates warning
        # self.assertEqual(est.test_conditional_independence('Sex', 'Survived', ["Age", "Embarked"]),
        #                 (235.51133052530713, 0.99999999683394869, False))
