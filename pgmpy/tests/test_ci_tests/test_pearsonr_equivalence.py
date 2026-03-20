import os
import unittest

import numpy as np

from pgmpy.ci_tests import PearsonrEquivalence
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@unittest.skipIf(os.getenv("GITHUB_ACTIONS") == "true", "Skipping residual tests on GitHub Actions.")
class TestPearsonrEquivalence(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

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
        cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
        cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
        cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
        cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
        cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
        model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
        self.df_dep = model_dep.simulate(n_samples=1000, seed=42)

    def test_pearsonr_equivalence(self):
        test = PearsonrEquivalence(data=self.df_dep, delta_threshold=0.1)

        self.assertFalse(test("X", "Y", ["Z1", "Z2", "Z3"], significance_level=0.05))

        test("X", "Y", ["Z1", "Z2", "Z3"])
        self.assertIsInstance(test.statistic_, float)
        self.assertIsInstance(test.p_value_, float)
