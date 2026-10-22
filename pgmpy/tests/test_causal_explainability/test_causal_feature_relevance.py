import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_explainability.causal_feature_relevance import CausalFeatureRelevance
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestCausalFeatureRelevanceLinearGaussian(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X1", "Y"), ("X2", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X1", beta=[0.0], std=1.0),
            LinearGaussianCPD("X2", beta=[0.0], std=2.0),
            LinearGaussianCPD("Y", beta=[0.0, 3.0, 1.0], std=0.5, evidence=["X1", "X2"]),
        )
        rng = np.random.default_rng(42)
        n = 5000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 2, n)
        y = 3.0 * x1 + 1.0 * x2 + rng.normal(0, 0.5, n)
        self.data = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    def test_analytical_variance(self):
        cfr = CausalFeatureRelevance()
        result = cfr.attribute(self.model, self.data, target="Y")
        # beta_1^2 * Var(X1) = 9 * 1 = 9, beta_2^2 * Var(X2) = 1 * 4 = 4
        self.assertAlmostEqual(result["X1"], 9.0, places=0)
        self.assertAlmostEqual(result["X2"], 4.0, places=0)

    def test_efficiency(self):
        cfr = CausalFeatureRelevance()
        result = cfr.attribute(self.model, self.data, target="Y")
        self.assertAlmostEqual(sum(result.values()), 13.0, places=0)

    def test_node_with_no_parents(self):
        cfr = CausalFeatureRelevance()
        result = cfr.attribute(self.model, self.data, target="X1")
        self.assertEqual(result, {})

    def test_graph_level(self):
        cfr = CausalFeatureRelevance(level="graph")
        result = cfr.attribute(self.model, self.data, target=None)
        self.assertIn("Y", result)
        self.assertIn("X1", result)
        self.assertEqual(result["X1"], {})
        self.assertAlmostEqual(result["Y"]["X1"], 9.0, places=0)
