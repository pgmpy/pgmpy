import unittest

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import TabularCPD


class TestMLE(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianModel([('A', 'C'), ('B', 'C')])
        self.d1 = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})

    def test_get_parameters_missing_data(self):
        mle = MaximumLikelihoodEstimator(self.m1, self.d1)
        cpds = [TabularCPD('A', 2, [[2.0/3], [1.0/3]]),
                TabularCPD('C', 2, [[0.0, 0.0, 1.0, 0.5],
                                    [1.0, 1.0, 0.0, 0.5]],
                           evidence=['A', 'B'], evidence_card=[2, 2]),
                TabularCPD('B', 2, [[2.0/3], [1.0/3]])]

        self.assertSetEqual(set(mle.get_parameters()), set(cpds))

    def tearDown(self):
        del self.m1
        del self.d1
