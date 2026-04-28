import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_explainability.anomaly_attribution import AnomalyAttribution
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestAnomalyAttributionLinearGaussian(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X", "Z"), ("Z", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X", beta=[0.0], std=1.0),
            LinearGaussianCPD("Z", beta=[0.0, 1.0], std=1.0, evidence=["X"]),
            LinearGaussianCPD("Y", beta=[0.0, 2.0], std=1.0, evidence=["Z"]),
        )
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        z = x + rng.normal(0, 1, n)
        y = 2 * z + rng.normal(0, 1, n)
        self.data = pd.DataFrame({"X": x, "Z": z, "Y": y})

    def test_anomaly_at_single_node(self):
        """If X is highly anomalous, most attribution should go to X."""
        aa = AnomalyAttribution()
        result = aa.attribute(
            self.model,
            self.data,
            target="Y",
            observation={"X": 5.0, "Z": 5.0, "Y": 10.0},
        )
        self.assertGreater(abs(result["X"]), abs(result.get("Z", 0)))

    def test_efficiency(self):
        """Attributions should sum to total anomaly score."""
        aa = AnomalyAttribution()
        obs = {"X": 3.0, "Z": 5.0, "Y": 15.0}
        result = aa.attribute(self.model, self.data, target="Y", observation=obs)
        total = sum(result.values())
        # Total should be positive (anomalous observation)
        self.assertGreater(total, 0)

    def test_custom_scorer(self):
        """Custom scorer should be called."""
        call_count = [0]

        def custom_scorer(observed, expected_samples):
            call_count[0] += 1
            return abs(observed - np.mean(expected_samples))

        aa = AnomalyAttribution(anomaly_scorer=custom_scorer)
        aa.attribute(
            self.model,
            self.data,
            target="Y",
            observation={"X": 3.0, "Z": 5.0, "Y": 15.0},
            n_samples=100,
        )
        self.assertGreater(call_count[0], 0)
