import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_explainability.intrinsic_causal_influence import IntrinsicCausalInfluence
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestIntrinsicCausalInfluenceLinearGaussian(unittest.TestCase):
    def setUp(self):
        # X1 -> X2 -> Y, X1 -> Y
        self.model = LinearGaussianBayesianNetwork([("X1", "X2"), ("X2", "Y"), ("X1", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X1", beta=[0.0], std=1.0),
            LinearGaussianCPD("X2", beta=[0.0, 2.0], std=1.0, evidence=["X1"]),
            LinearGaussianCPD("Y", beta=[0.0, 1.0, 3.0], std=0.5, evidence=["X1", "X2"]),
        )
        rng = np.random.default_rng(42)
        n = 5000
        n1 = rng.normal(0, 1, n)
        x1 = n1
        n2 = rng.normal(0, 1, n)
        x2 = 2.0 * x1 + n2
        n_y = rng.normal(0, 0.5, n)
        y = 1.0 * x1 + 3.0 * x2 + n_y
        self.data = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    def test_analytical_variance_decomposition(self):
        """Y = 1*X1 + 3*(2*X1 + N2) + N_Y = 7*N1 + 3*N2 + N_Y.
        Contributions: N1 -> 49, N2 -> 9, N_Y -> 0.25"""
        ici = IntrinsicCausalInfluence()
        result = ici.attribute(self.model, self.data, target="Y")
        self.assertAlmostEqual(result["X1"], 49.0, places=5)
        self.assertAlmostEqual(result["X2"], 9.0, places=5)
        self.assertAlmostEqual(result["Y"], 0.25, places=5)

    def test_efficiency(self):
        ici = IntrinsicCausalInfluence()
        result = ici.attribute(self.model, self.data, target="Y")
        self.assertAlmostEqual(sum(result.values()), 58.25, places=5)

    def test_single_node_chain(self):
        model = LinearGaussianBayesianNetwork([("X", "Y")])
        model.add_cpds(
            LinearGaussianCPD("X", beta=[0.0], std=2.0),
            LinearGaussianCPD("Y", beta=[0.0, 3.0], std=1.0, evidence=["X"]),
        )
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 2, n)
        y = 3 * x + rng.normal(0, 1, n)
        data = pd.DataFrame({"X": x, "Y": y})
        ici = IntrinsicCausalInfluence()
        result = ici.attribute(model, data, target="Y")
        # w_X = 3, sigma_X = 2: 9 * 4 = 36
        # w_Y = 1, sigma_Y = 1: 1 * 1 = 1
        self.assertAlmostEqual(result["X"], 36.0, places=5)
        self.assertAlmostEqual(result["Y"], 1.0, places=5)

    def test_root_node_only_self_noise(self):
        ici = IntrinsicCausalInfluence()
        result = ici.attribute(self.model, self.data, target="X1")
        self.assertAlmostEqual(result["X1"], 1.0, places=5)
