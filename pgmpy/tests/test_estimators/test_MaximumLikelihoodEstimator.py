import unittest

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import TabularCPD


class TestMLE(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianModel([('A', 'C'), ('B', 'C')])
        self.d1 = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        self.cpds = [TabularCPD('A', 2, [[2.0/3], [1.0/3]]),
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

    def test_state_names1(self):
        m = BayesianModel([('A', 'B')])
        d = pd.DataFrame(data={'A': [2, 3, 8, 8, 8], 'B': ['X', 'O', 'X', 'O', 'X']})
        cpd_b = TabularCPD('B', 2, [[0, 1, 1.0 / 3], [1, 0, 2.0 / 3]],
                           evidence=['A'], evidence_card=[3])
        mle2 = MaximumLikelihoodEstimator(m, d)
        self.assertEqual(mle2._estimate_cpd('B'), cpd_b)

    def test_state_names2(self):
        m = BayesianModel([('Light?', 'Color'), ('Fruit', 'Color')])
        d = pd.DataFrame(data={'Fruit': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana'],
                               'Light?': [True,   True,   False,   False,    True],
                               'Color': ['red',   'green', 'black', 'black',  'yellow']})
        color_cpd = TabularCPD('Color', 4, [[1, 0, 1, 0], [0, 0.5, 0, 0],
                                            [0, 0.5, 0, 0], [0, 0, 0, 1]],
                               evidence=['Fruit', 'Light?'], evidence_card=[2, 2])
        mle2 = MaximumLikelihoodEstimator(m, d)
        self.assertEqual(mle2._estimate_cpd('Color'), color_cpd)

    def test_class_init(self):
        mle = MaximumLikelihoodEstimator(self.m1, self.d1,
                                         node_values={'A': [0, 1], 'B': [0, 1], 'C': [0, 1]})
        self.assertSetEqual(set(mle.get_parameters()), set(self.cpds))

    def test_nonoccurring_values(self):
        mle = MaximumLikelihoodEstimator(self.m1, self.d1,
                                         node_values={'A': [0, 1, 23], 'B': [0, 1], 'C': [0, 42, 1], 1: [2]})
        cpds = [TabularCPD('A', 3, [[2.0/3], [1.0/3], [0]]),
                TabularCPD('B', 2, [[2.0/3], [1.0/3]]),
                TabularCPD('C', 3, [[0.0, 0.0, 1.0, 1.0/3, 1.0/3, 1.0/3],
                                    [1.0, 1.0, 0.0, 1.0/3, 1.0/3, 1.0/3],
                                    [0.0, 0.0, 0.0, 1.0/3, 1.0/3, 1.0/3]],
                           evidence=['A', 'B'], evidence_card=[3, 2])]
        self.assertSetEqual(set(mle.get_parameters()), set(cpds))

    def tearDown(self):
        del self.m1
        del self.d1
