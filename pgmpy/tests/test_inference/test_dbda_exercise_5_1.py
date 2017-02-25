#!/usr/bin/env python3
import unittest
import numpy as np
import pandas as pd
from pprint import pprint
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from collections import defaultdict

# The test case follows exercise 5.1 from "Doing Bayesian Data Analysis" by John K. Kruschke:
# https://sites.google.com/site/doingbayesiandataanalysis/exercises

p_disease_present = 0.001
prior = pd.DataFrame([p_disease_present, 1-p_disease_present], columns=['disease-state'])
prior.index = pd.Index(['disease-present', 'disease-absent'])

p_test_positive_given_disease_present = 0.99
p_test_positive_given_disease_absent = 0.05

disease_test_cpd_df = pd.DataFrame([[p_test_positive_given_disease_present, p_test_positive_given_disease_absent],
                                    [1 - p_test_positive_given_disease_present, 1 - p_test_positive_given_disease_absent]], columns=['disease-present', 'disease-absent'])
disease_test_cpd_df.index = pd.Index(['test-positive', 'test-negative'])

posterior = prior['disease-state'].copy()

selected_row = disease_test_cpd_df.loc['test-positive', :]
posterior = selected_row * posterior
posterior = posterior / float(posterior.sum())
posterior['disease-present']

for x in ['test-positive', 'test-negative']:
    selected_row = disease_test_cpd_df.loc[x,:]
    posterior = selected_row * posterior
    posterior = posterior / float(posterior.sum())


class TestDBDAExercise51(unittest.TestCase):

    def setUp(self):
        self.model = BayesianModel()
        self.model.add_nodes_from(['disease-state', 'test-result1', 'test-result2'])
        self.model.add_edge('disease-state', 'test-result1')
        self.model.add_edge('disease-state', 'test-result2')

        p_disease_present = 0.001
        disease_state_CPD = TabularCPD(variable='disease-state',
                                       variable_card=2,
                                       values=[[p_disease_present], [1.0 - p_disease_present]])
        p_test_positive_given_disease_present = 0.99
        p_test_positive_given_disease_absent = 0.05

        test_result_CPD_1 = TabularCPD(variable='test-result1',
                                       variable_card=2,
                                       values=[[p_test_positive_given_disease_present, p_test_positive_given_disease_absent],
                                               [(1 - p_test_positive_given_disease_present), (1 - p_test_positive_given_disease_absent)]],
                                       evidence=['disease-state'],
                                       evidence_card=[2])

        test_result_CPD_2 = TabularCPD(variable='test-result2',
                                       variable_card=2,
                                       values=[[p_test_positive_given_disease_present, p_test_positive_given_disease_absent],
                                               [(1 - p_test_positive_given_disease_present), (1 - p_test_positive_given_disease_absent)]],
                                       evidence=['disease-state'],
                                       evidence_card=[2])

        self.model.add_cpds(disease_state_CPD, test_result_CPD_1, test_result_CPD_2)
        self.model.check_model()

        self.prior = pd.DataFrame([p_disease_present, 1 - p_disease_present], columns=['disease-state'])
        self.prior.index = pd.Index(['disease-present', 'disease-absent'])
        self.disease_test_cpd_df = pd.DataFrame([[p_test_positive_given_disease_present, p_test_positive_given_disease_absent],
                                                 [1 - p_test_positive_given_disease_present, 1 - p_test_positive_given_disease_absent]],
                                                columns=['disease-present', 'disease-absent'])
        self.disease_test_cpd_df.index = pd.Index(['test-positive', 'test-negative'])

    def calculate_posterior(self, input=['test-positive', 'test-negative']):
        posterior = self.prior['disease-state'].copy()
        for x in input:
            selected_row = self.disease_test_cpd_df.loc[x, :]
            posterior = selected_row * posterior
            posterior = posterior / float(posterior.sum())
        return posterior

    def test_integration_one_positive(self):
        posterior = self.calculate_posterior(input=['test-positive')
        p_disease = posterior['disease-present']

        pprint('{0:f}'.format(p_disease))
        self.assertAlmostEqual(p_disease, 0.01943463, places=7)

    def test_one_positive(self):
        infr1 = VariableElimination(self.model)

        evidence = {'test-result1': 0}
        query_vars = ['disease-state']
        p_disease = infr1.query(variables=query_vars, evidence=evidence)['disease-state'].values[0]

        pprint('{0:f}'.format(p_disease))
        self.assertAlmostEqual(p_disease, 0.01943463, places=7)

    def test_one_positive_and_one_negative(self):
        infr1 = VariableElimination(self.model)

        evidence = {'test-result1': 0, 'test-result2': 1}
        query_vars = ['disease-state']
        p_disease = infr1.query(variables=query_vars, evidence=evidence)['disease-state'].values[0]

        pprint('{0:f}'.format(p_disease))
        self.assertAlmostEqual(p_disease, 0.0002085862, places=7)

    def test_two_positive(self):
        infr1 = VariableElimination(self.model)

        evidence = {'test-result1': 0, 'test-result2': 0}
        query_vars = ['disease-state']
        p_disease = infr1.query(variables=query_vars, evidence=evidence)['disease-state'].values[0]

        pprint('{0:f}'.format(p_disease))
        #self.assertAlmostEqual(p_disease, 0.0002085862, places=7)

if __name__ == '__main__':
    unittest.main()
