import os
import unittest

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.ci_tests import PillaiTrace
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping residual tests on GitHub Actions.")
class TestPillaiTrace(unittest.TestCase):
    def setUp(self):
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

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_no_cond(self):
        computed_coefs = []
        computed_pvalues = []
        for i, df in enumerate(
            [
                self.df_indep,
                self.df_indep_cont_cont,
                self.df_indep_cat_cont,
                self.df_indep_cat_cat,
                self.df_indep_ord_cont,
            ]
        ):
            test = PillaiTrace(data=df, seed=42)
            test("X", "Y", [])
            computed_coefs.append(test.statistic_)
            computed_pvalues.append(test.p_value_)

        self.assertTrue(np.all(np.array(computed_coefs) >= 0.1))
        self.assertTrue(np.all(np.array(computed_pvalues) <= 0.05))

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_indep(self):
        computed_coefs = []
        computed_pvalues = []
        for i, df in enumerate(
            [
                self.df_indep,
                self.df_indep_cont_cont,
                self.df_indep_cat_cont,
                self.df_indep_cat_cat,
                self.df_indep_ord_cont,
            ]
        ):
            test = PillaiTrace(data=df, seed=42)
            test("X", "Y", ["Z1", "Z2", "Z3"])
            computed_coefs.append(test.statistic_)
            computed_pvalues.append(test.p_value_)

        self.assertTrue(np.all(np.array(computed_coefs) <= 0.1))
        self.assertTrue(np.all(np.array(computed_pvalues) >= 0.05))

    @unittest.skipUnless(
        _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_pillai_dependent(self):
        computed_coefs = []
        computed_pvalues = []
        for i, df in enumerate(
            [
                self.df_dep,
                self.df_dep_cont_cont,
                self.df_dep_cat_cont,
                self.df_dep_cat_cat,
                self.df_dep_ord_cont,
            ]
        ):
            test = PillaiTrace(data=df, seed=42)
            test("X", "Y", ["Z1", "Z2", "Z3"])
            computed_coefs.append(test.statistic_)
            computed_pvalues.append(test.p_value_)

        self.assertTrue(np.all(np.array(computed_coefs) >= 0.1))
        self.assertTrue(np.all(np.array(computed_pvalues) <= 0.05))
