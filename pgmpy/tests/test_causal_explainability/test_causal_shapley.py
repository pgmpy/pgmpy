import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_explainability.causal_shapley import CausalShapleyValues
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestCausalShapleyLinearGaussian(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X1", "Y"), ("X2", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X1", beta=[0.0], std=1.0),
            LinearGaussianCPD("X2", beta=[0.0], std=1.0),
            LinearGaussianCPD("Y", beta=[0.0, 2.0, 3.0], std=0.5, evidence=["X1", "X2"]),
        )
        rng = np.random.default_rng(42)
        n = 5000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2.0 * x1 + 3.0 * x2 + rng.normal(0, 0.5, n)
        self.data = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    def test_interventional_efficiency(self):
        csv = CausalShapleyValues(method="interventional")
        obs = {"X1": 2.0, "X2": 1.0}
        result = csv.attribute(self.model, self.data, target="Y", observation=obs)
        f_obs = 2.0 * 2.0 + 3.0 * 1.0
        f_mean = 2.0 * self.data["X1"].mean() + 3.0 * self.data["X2"].mean()
        self.assertAlmostEqual(sum(result.values()), f_obs - f_mean, places=1)

    def test_interventional_independent_causes(self):
        csv = CausalShapleyValues(method="interventional")
        obs = {"X1": 2.0, "X2": 1.0}
        result = csv.attribute(self.model, self.data, target="Y", observation=obs)
        expected_x1 = 2.0 * (2.0 - self.data["X1"].mean())
        expected_x2 = 3.0 * (1.0 - self.data["X2"].mean())
        self.assertAlmostEqual(result["X1"], expected_x1, places=1)
        self.assertAlmostEqual(result["X2"], expected_x2, places=1)

    def test_observational_efficiency(self):
        csv = CausalShapleyValues(method="observational")
        obs = {"X1": 2.0, "X2": 1.0}
        result = csv.attribute(self.model, self.data, target="Y", observation=obs)
        f_obs = 2.0 * 2.0 + 3.0 * 1.0
        f_mean = 2.0 * self.data["X1"].mean() + 3.0 * self.data["X2"].mean()
        self.assertAlmostEqual(sum(result.values()), f_obs - f_mean, places=1)

    def test_causal_method_efficiency(self):
        csv = CausalShapleyValues(method="causal")
        obs = {"X1": 2.0, "X2": 1.0}
        result = csv.attribute(self.model, self.data, target="Y", observation=obs)
        f_obs = 2.0 * 2.0 + 3.0 * 1.0
        f_mean = 2.0 * self.data["X1"].mean() + 3.0 * self.data["X2"].mean()
        self.assertAlmostEqual(sum(result.values()), f_obs - f_mean, places=1)


class TestCausalShapleyConfounded(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X1", "X2"), ("X2", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X1", beta=[0.0], std=1.0),
            LinearGaussianCPD("X2", beta=[0.0, 1.0], std=1.0, evidence=["X1"]),
            LinearGaussianCPD("Y", beta=[0.0, 2.0], std=0.5, evidence=["X2"]),
        )
        rng = np.random.default_rng(42)
        n = 5000
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 1, n)
        y = 2.0 * x2 + rng.normal(0, 0.5, n)
        self.data = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    def test_interventional_vs_observational_differ(self):
        obs = {"X1": 2.0, "X2": 3.0}
        result_int = CausalShapleyValues(method="interventional").attribute(
            self.model, self.data, target="Y", observation=obs
        )
        result_obs = CausalShapleyValues(method="observational").attribute(
            self.model, self.data, target="Y", observation=obs
        )
        self.assertNotAlmostEqual(result_int["X1"], result_obs["X1"], places=0)
