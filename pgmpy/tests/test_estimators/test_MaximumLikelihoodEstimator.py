import unittest

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import TabularCPD


class TestMLE(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianModel([('A', 'C'), ('B', 'C')])
        self.d1 = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        self.cpds = cpds = [TabularCPD('A', 2, [[2.0/3], [1.0/3]]),
                            TabularCPD('B', 2, [[2.0/3], [1.0/3]]),
                            TabularCPD('C', 2, [[0.0, 0.0, 1.0, 0.5],
                                                [1.0, 1.0, 0.0, 0.5]],
                                       evidence=['A', 'B'], evidence_card=[2, 2])]
        self.mle1 = MaximumLikelihoodEstimator(self.m1, self.d1)

    def test_get_parameters_missing_data(self):
        self.assertSetEqual(set(self.mle1.get_parameters()), set(self.cpds))

    def test_estimate_cpd(self):
        self.assertEqual(self.mle1._estimate_cpd('A'), self.cpds[0])
        self.assertEqual(self.mle1._estimate_cpd('B'), self.cpds[1])
        self.assertEqual(self.mle1._estimate_cpd('C'), self.cpds[2])

    def tearDown(self):
        del self.m1
        del self.d1
