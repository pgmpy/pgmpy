import unittest

import numpy as np
import pandas as pd

from pgmpy.causal_explainability._base import _BaseAttribution
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestBaseAttribution(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X", beta=[0.0], std=1.0),
            LinearGaussianCPD("Y", beta=[0.0, 0.5], std=1.0, evidence=["X"]),
        )
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        y = 0.5 * x + rng.normal(0, 1, n)
        self.data = pd.DataFrame({"X": x, "Y": y})

    def test_attribute_raises_not_implemented(self):
        base = _BaseAttribution()
        with self.assertRaises(NotImplementedError):
            base.attribute(self.model, self.data, target="Y")

    def test_validate_target_not_in_model(self):
        base = _BaseAttribution()
        with self.assertRaises(ValueError):
            base.attribute(self.model, self.data, target="Z")

    def test_validate_data_missing_columns(self):
        base = _BaseAttribution()
        bad_data = pd.DataFrame({"X": [1, 2], "W": [3, 4]})
        with self.assertRaises(ValueError):
            base.attribute(self.model, bad_data, target="Y")

    def test_validate_data_not_dataframe(self):
        base = _BaseAttribution()
        with self.assertRaises(ValueError):
            base.attribute(self.model, [[1, 2], [3, 4]], target="Y")
