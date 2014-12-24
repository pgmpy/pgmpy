from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
import unittest
import numpy as np
from numpy import testing


class TestVariableElimination(unittest.TestCase):
    def setUp(self):
        self.bayesian_model = BayesianModel([('A', 'J'), ('R', 'J'), ('J', 'Q'),
                                             ('J', 'L'), ('G', 'L')])
        cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
        cpd_j = TabularCPD('J', 2,
                           [[0.9, 0.6, 0.7, 0.1],
                            [0.1, 0.4, 0.3, 0.9]],
                           ['R', 'A'], [2, 2])
        cpd_q = TabularCPD('Q', 2,
                           [[0.9, 0.2],
                            [0.1, 0.8]],
                           ['J'], [2])
        cpd_l = TabularCPD('L', 2,
                           [[0.9, 0.45, 0.8, 0.1],
                            [0.1, 0.55, 0.2, 0.9]],
                           ['G', 'J'], [2, 2])
        cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)

        self.bayesian_inference = VariableElimination(self.bayesian_model)

    # All the values that are used for comparision in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        query_result = self.bayesian_inference.query(['J'])
        testing.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))

    def test_query_multiple_variable(self):
        query_result = self.bayesian_inference.query(['Q', 'J'])
        testing.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))
        testing.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.4912, 0.5088]))

    def test_query_single_variable_with_evidence(self):
        query_result = self.bayesian_inference.query(variables=['J'],
                                                     evidence={'A': 0, 'R': 1})
        testing.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.60, 0.40]))

    def test_query_multiple_variable_with_evidence(self):
        query_result = self.bayesian_inference.query(variables=['J', 'Q'],
                                                     evidence={'A': 0, 'R': 0,
                                                               'G': 0, 'L': 1})
        testing.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.818182, 0.181818]))
        testing.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.772727, 0.227273]))

