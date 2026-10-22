import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_explainability.distribution_change import DistributionChangeAttribution
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestDistributionChangeLinearGaussian(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X", beta=[0.0], std=1.0),
            LinearGaussianCPD("Y", beta=[0.0, 2.0], std=1.0, evidence=["X"]),
        )
        rng = np.random.default_rng(42)
        n = 2000
        x_old = rng.normal(0, 1, n)
        y_old = 2.0 * x_old + rng.normal(0, 1, n)
        self.data_old = pd.DataFrame({"X": x_old, "Y": y_old})
        x_new = rng.normal(3.0, 1, n)
        y_new = 2.0 * x_new + rng.normal(0, 1, n)
        self.data_new = pd.DataFrame({"X": x_new, "Y": y_new})

    def test_change_attributed_to_changed_mechanism(self):
        dca = DistributionChangeAttribution()
        result = dca.attribute(self.model, data_old=self.data_old, data_new=self.data_new, target="Y", n_samples=500)
        self.assertGreater(result["X"], result["Y"])

    def test_no_change(self):
        dca = DistributionChangeAttribution()
        result = dca.attribute(self.model, data_old=self.data_old, data_new=self.data_old, target="Y", n_samples=500)
        for v in result.values():
            self.assertAlmostEqual(v, 0.0, places=0)

    def test_mechanism_change_test(self):
        dca = DistributionChangeAttribution()
        pvals = dca.mechanism_change_test(self.model, self.data_old, self.data_new)
        self.assertIn("X", pvals)
        self.assertIn("Y", pvals)
        self.assertLess(pvals["X"], 0.05)
