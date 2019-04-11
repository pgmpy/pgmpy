import unittest

import numpy as np
import pandas as pd

from pgmpy.models.CausalGraph import CausalGraph
from pgmpy.inference.CausalInference import CausalInference


class TestEstimator(unittest.TestCase):

    def test_create_estimator(self):
        game1 = CausalGraph([('X', 'A'),
                               ('A', 'Y'),
                               ('A', 'B')])
        data = pd.DataFrame(np.random.randint(2, size=(1000, 4)), columns=['X', 'A', 'B', 'Y'])
        inference = CausalInference(model=game1)
        ate = inference.estimate_ate("X", "Y", data=data, estimator_type="linear")
        self.assertAlmostEqual(ate, 0, places=0)
