import unittest

import pandas as pd
import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD


class TestBayesianEstimator(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianModel([('A', 'C'), ('B', 'C')])
        self.d1 = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        self.d2 = pd.DataFrame(data={'A': [0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
                                     'B': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
                                     'C': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]})
        self.est1 = BayesianEstimator(self.m1, self.d1)
        self.est2 = BayesianEstimator(self.m1, self.d1, state_names={'A': [0, 1, 2],
                                                                     'B': [0, 1],
                                                                     'C': [0, 1, 23]})
        self.est3 = BayesianEstimator(self.m1, self.d2)

    def test_estimate_cpd_dirichlet(self):
        cpd_A = self.est1.estimate_cpd('A',  prior_type="dirichlet", pseudo_counts=[0, 1])
        self.assertEqual(cpd_A, TabularCPD('A', 2, [[0.5], [0.5]]))

        cpd_B = self.est1.estimate_cpd('B',  prior_type="dirichlet", pseudo_counts=[9, 3])
        self.assertEqual(cpd_B, TabularCPD('B', 2, [[11.0/15], [4.0/15]]))

        cpd_C = self.est1.estimate_cpd('C',  prior_type="dirichlet", pseudo_counts=[0.4, 0.6])
        self.assertEqual(cpd_C, TabularCPD('C', 2, [[0.2, 0.2, 0.7, 0.4],
                                                    [0.8, 0.8, 0.3, 0.6]],
                                           evidence=['A', 'B'], evidence_card=[2, 2]))

    def test_estimate_cpd_improper_prior(self):
        cpd_C = self.est1.estimate_cpd('C',  prior_type="dirichlet", pseudo_counts=[0, 0])
        cpd_C_correct = (TabularCPD('C', 2, [[0.0, 0.0, 1.0, np.NaN],
                                             [1.0, 1.0, 0.0, np.NaN]],
                                    evidence=['A', 'B'], evidence_card=[2, 2],
                                    state_names={'A': [0, 1], 'B': [0, 1], 'C': [0, 1]}))
        # manual comparison because np.NaN != np.NaN
        self.assertTrue(((cpd_C.values == cpd_C_correct.values) |
                         np.isnan(cpd_C.values) & np.isnan(cpd_C_correct.values)).all())

    def test_estimate_cpd_shortcuts(self):
        cpd_C1 = self.est2.estimate_cpd('C',  prior_type='BDeu', equivalent_sample_size=9)
        cpd_C1_correct = TabularCPD('C', 3, [[0.2, 0.2, 0.6, 1./3, 1./3, 1./3],
                                             [0.6, 0.6, 0.2, 1./3, 1./3, 1./3],
                                             [0.2, 0.2, 0.2, 1./3, 1./3, 1./3]],
                                    evidence=['A', 'B'], evidence_card=[3, 2])
        self.assertEqual(cpd_C1, cpd_C1_correct)

        cpd_C2 = self.est3.estimate_cpd('C',  prior_type='K2')
        cpd_C2_correct = TabularCPD('C', 2, [[0.5, 0.6, 1./3, 2./3, 0.75, 2./3],
                                             [0.5, 0.4, 2./3, 1./3, 0.25, 1./3]],
                                    evidence=['A', 'B'], evidence_card=[3, 2])
        self.assertEqual(cpd_C2, cpd_C2_correct)

    def test_get_parameters(self):
        cpds = set([self.est3.estimate_cpd('A'),
                    self.est3.estimate_cpd('B'),
                    self.est3.estimate_cpd('C')])
        self.assertSetEqual(set(self.est3.get_parameters()), cpds)

    def test_get_parameters2(self):
        pseudo_counts = {'A': [1, 2, 3], 'B': [4, 5], 'C': [6, 7]}
        cpds = set([self.est3.estimate_cpd('A', prior_type="dirichlet", pseudo_counts=pseudo_counts['A']),
                    self.est3.estimate_cpd('B', prior_type="dirichlet", pseudo_counts=pseudo_counts['B']),
                    self.est3.estimate_cpd('C', prior_type="dirichlet", pseudo_counts=pseudo_counts['C'])])
        self.assertSetEqual(set(self.est3.get_parameters(prior_type="dirichlet",
                                                         pseudo_counts=pseudo_counts)), cpds)

    def tearDown(self):
        del self.m1
        del self.d1
        del self.d2
        del self.est1
        del self.est2
