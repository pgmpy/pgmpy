import os
import unittest

from sklearn.ensemble import RandomForestRegressor

from pgmpy.ci_tests import GCM
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestGCM(unittest.TestCase):
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
        self.df_indep = model_indep.simulate(n_samples=10000, seed=42)

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
        self.df_dep = model_dep.simulate(n_samples=10000, seed=42)

    @unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping exact residual tests on GitHub Actions.")
    def test_gcm_exact(self):
        test = GCM(data=self.df_indep)

        # Non-conditional test
        test("X", "Y", [])
        self.assertAlmostEqual(round(test.statistic_, 3), 39.631)
        self.assertAlmostEqual(test.p_value_, 0.0)

        # Conditional test (independent)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 0.584)
        self.assertEqual(round(test.p_value_, 4), 0.5591)

        # Conditional test (dependent)
        test = GCM(data=self.df_dep)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertAlmostEqual(round(test.statistic_, 3), 39.798)
        self.assertAlmostEqual(test.p_value_, 0.0)

        # Test with custom sklearn estimator
        test = GCM(data=self.df_indep, estimator=RandomForestRegressor(random_state=42))
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertIsInstance(test.statistic_, float)
        self.assertIsInstance(test.p_value_, float)
        self.assertGreaterEqual(test.p_value_, 0.0)
        self.assertLessEqual(test.p_value_, 1.0)

    def test_gcm_approx(self):
        test = GCM(data=self.df_indep)

        # Non-conditional test
        test("X", "Y", [])
        self.assertTrue(test.statistic_ > 1)
        self.assertTrue(test.p_value_ < 0.05)

        # Conditional test (independent)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertTrue(test.statistic_ < 1)
        self.assertTrue(test.p_value_ > 0.05)

        # Conditional test (dependent)
        test = GCM(data=self.df_dep)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertTrue(test.statistic_ > 1)
        self.assertTrue(test.p_value_ < 0.05)

        # Test with custom sklearn estimator
        test = GCM(data=self.df_indep, estimator=RandomForestRegressor(random_state=42))
        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertTrue(test.p_value_ > 0.05)
