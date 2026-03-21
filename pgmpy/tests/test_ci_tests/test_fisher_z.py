import os
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from pgmpy.ci_tests import FisherZ, Pearsonr
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestFisherZ(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42)

        self.df_ind = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["X", "Y", "Z"])

        Z = rng.normal(10000)
        X = 0.3 * Z + rng.normal(loc=0, scale=0.1, size=10000)
        Y = 0.2 * Z + rng.normal(loc=0, scale=0.1, size=10000)
        self.df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        Z1 = rng.normal(10000)
        Z2 = rng.normal(10000)
        X = 0.3 * Z1 + 0.2 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
        Y = 0.2 * Z1 + 0.3 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
        self.df_cind_mul = pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})

        X = rng.normal(10000)
        Y = rng.normal(10000)
        Z = 0.2 * X + 0.2 * Y + rng.normal(loc=0, scale=0.1, size=10000)
        self.df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_fisher_z(self):
        test = FisherZ(data=self.df_ind)
        test("X", "Y", [])
        self.assertTrue(test.p_value_ > 0.05)

        test = FisherZ(data=self.df_cind)
        test("X", "Y", ["Z"])
        self.assertTrue(test.p_value_ > 0.05)

        test = FisherZ(data=self.df_cind_mul)
        test("X", "Y", ["Z1", "Z2"])
        self.assertTrue(test.p_value_ > 0.05)

        test = FisherZ(data=self.df_vstruct)
        test("X", "Y", ["Z"])
        self.assertTrue(test.p_value_ < 0.05)

        self.assertTrue(FisherZ(data=self.df_ind)("X", "Y", [], significance_level=0.05))
        self.assertTrue(FisherZ(data=self.df_cind)("X", "Y", ["Z"], significance_level=0.05))
        self.assertTrue(FisherZ(data=self.df_cind_mul)("X", "Y", ["Z1", "Z2"], significance_level=0.05))
        self.assertFalse(FisherZ(data=self.df_vstruct)("X", "Y", ["Z"], significance_level=0.05))

    def test_fisher_z_uses_pearsonr_partial_correlation(self):
        rng = np.random.default_rng(seed=7)
        n_samples = 40
        z_columns = [f"Z{i}" for i in range(5)]
        data = pd.DataFrame(rng.standard_normal((n_samples, len(z_columns))), columns=z_columns)
        data["X"] = data[z_columns].sum(axis=1) + rng.normal(scale=1.0, size=n_samples)
        data["Y"] = 0.35 * data["X"] + data[z_columns].sum(axis=1) + rng.normal(scale=1.2, size=n_samples)

        pearson_test = Pearsonr(data=data)
        pearson_test("X", "Y", z_columns)

        fisher_test = FisherZ(data=data)
        is_independent = fisher_test("X", "Y", z_columns, significance_level=0.05)

        expected_coeff = np.arctanh(np.clip(pearson_test.statistic_, -0.999999, 0.999999))
        expected_statistic = np.sqrt(n_samples - len(z_columns) - 3) * expected_coeff
        expected_p_value = 2 * stats.norm.sf(np.abs(expected_statistic))

        self.assertEqual(is_independent, expected_p_value >= 0.05)
        self.assertAlmostEqual(fisher_test.statistic_, expected_statistic)
        self.assertAlmostEqual(fisher_test.p_value_, expected_p_value)


@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping residual tests on GitHub Actions.")
class TestFisherZResidual(unittest.TestCase):
    def setUp(self):
        model_indep = LinearGaussianBayesianNetwork(
            [
                ("Z1", "X"),
                ("Z2", "X"),
                ("Z3", "X"),
                ("Z1", "Y"),
                ("Z2", "Y"),
                ("Z3", "Y"),
            ]
        )
        cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
        cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
        cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
        cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        cpd_y_indep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        model_indep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_indep)
        self.df_indep = model_indep.simulate(n_samples=1000, seed=42)

        model_dep = LinearGaussianBayesianNetwork(
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
        cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
        model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
        self.df_dep = model_dep.simulate(n_samples=1000, seed=42)

    def test_fisher_z(self):
        test = FisherZ(data=self.df_indep)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertTrue(test.p_value_ >= 0.05)

        test = FisherZ(data=self.df_dep)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertTrue(test.p_value_ <= 0.05)
