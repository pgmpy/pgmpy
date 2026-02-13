import os
import unittest

import numpy as np

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestLGBNIo(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("A", "B")])
        cpd_a = LinearGaussianCPD(variable="A", beta=[1.0], std=2.0)
        cpd_b = LinearGaussianCPD(
            variable="B", beta=[-5.0, 0.5], std=1.73205, evidence=["A"]
        )
        self.model.add_cpds(cpd_a, cpd_b)
        self.filename = "test_lgbn.json"

    def test_save_and_load(self):
        self.model.save(self.filename)

        model_loaded = LinearGaussianBayesianNetwork.load(self.filename)

        self.assertCountEqual(self.model.nodes(), model_loaded.nodes())
        self.assertCountEqual(self.model.edges(), model_loaded.edges())

        for node in self.model.nodes():
            original_cpd = self.model.get_cpds(node)
            loaded_cpd = model_loaded.get_cpds(node)

            np.testing.assert_allclose(original_cpd.beta, loaded_cpd.beta, rtol=1e-5)
            np.testing.assert_allclose(
                original_cpd.std**2, loaded_cpd.std**2, rtol=1e-5
            )

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
