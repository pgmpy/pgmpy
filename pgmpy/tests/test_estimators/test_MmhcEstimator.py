import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import MmhcEstimator, K2Score
from pgmpy.factors import TabularCPD
from pgmpy.extern import six
from pgmpy.models import BayesianModel


class TestMmhcEstimator(unittest.TestCase):
    def setUp(self):
        self.data1 = pd.DataFrame(np.random.randint(0, 2, size=(1500, 3)), columns=list('XYZ'))
        self.data1['sum'] = self.data1.sum(axis=1)
        self.est1 = MmhcEstimator(self.data1)

    def test_estimate(self):
        self.assertSetEqual(set(self.est1.estimate().edges()),
                            set([('X', 'sum'), ('Y', 'sum'), ('Z', 'sum')]))
        self.assertSetEqual(set(self.est1.estimate(significance_level=0.001).edges()),
                            set([('X', 'sum'), ('Y', 'sum'), ('Z', 'sum')]))

    def tearDown(self):
        del self.data1
        del self.est1
