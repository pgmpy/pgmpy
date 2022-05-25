import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TestEMObserved(unittest.TestCase):
    def setUp(self):
        self.model1 = get_example_model("cancer")
        s = BayesianModelSampling(self.model1)
        self.data1 = s.forward_sample(int(1e4))

        self.model2 = BayesianNetwork(self.model1.edges(), latents={"Smoker"})
        self.model2.add_cpds(*self.model1.cpds)
        s = BayesianModelSampling(self.model2)
        self.data2 = s.forward_sample(int(1e4))

    def test_get_parameters(self):
        est = EM(self.model1, self.data1)
        cpds = est.get_parameters(seed=42, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(seed=42, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [0, 1]

            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.3))
