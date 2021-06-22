import unittest

import pandas as pd
import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.utils import get_example_model


class TestEMObserved(unittest.TestCase):
    def setUp(self):
        self.model = get_example_model('alarm')
        s = BayesianModelSampling(self.model)
        self.data = s.forward_sample(int(1e4))

    def test_get_parameters(self):
        est = EM(self.model, self.data)
        cpds = est.get_parameters()
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model.get_cpds(var)
            import pdb; pdb.set_trace()
            self.assertTrue(orig_cpd.__eq__(est_cpd))
