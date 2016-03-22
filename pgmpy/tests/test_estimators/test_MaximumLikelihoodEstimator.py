import unittest

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import TabularCPD


class TestMLE(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianModel([('A','C'),('B','C')])
        self.m2 = BayesianModel([('A','B'),('B','C')])
        self.m3 = BayesianModel([('B','A'),('B','C')])
        self.d1 = pd.DataFrame(data={'A':[0,0,1], 'B':[0,1,0], 'C':[1,1,0]})
        self.d2 = pd.DataFrame(data={'A':[0,0,1,1], 'B':[0,1,0,1], 'C':[1,1,0,0]})


    def test_get_parameters_causal(self):
        mle = MaximumLikelihoodEstimator(self.m2, self.d2)
        cpds = [ TabularCPD('A', 2, [[0.5],[0.5]]),
                 TabularCPD('B', 2, [[0.5, 0.5],
                                     [0.5, 0.5]],
                                     evidence=['A'], evidence_card=[2]),
                 TabularCPD('C', 2, [[0.5, 0.5],
                                     [0.5, 0.5]],
                                     evidence=['B'], evidence_card=[2])]

        self.assertEqual(set(mle.get_parameters()), set(cpds))


    def test_get_parameters_common_cause(self):
        mle = MaximumLikelihoodEstimator(self.m3, self.d2)
        cpds = [ TabularCPD('B', 2, [[0.5],[0.5]]),
                 TabularCPD('C', 2, [[0.5, 0.5],
                                     [0.5, 0.5]],
                                     evidence=['B'], evidence_card=[2]),
                 TabularCPD('A', 2, [[0.5, 0.5],
                                     [0.5, 0.5]],
                                     evidence=['B'], evidence_card=[2])]

        self.assertEqual(set(mle.get_parameters()), set(cpds))


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
        del self.m2
        del self.m3

        del self.d1
        del self.d2
