import unittest
import numpy as np
import numpy.testing as np_test
from pgmpy.inference import DBNInference
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors import TabularCPD

# The sample Dynamic Bayesian Network is taken from the following paper:-
# Novel recursive inference algorithm for discrete dynamic Bayesian networks
# Huange Wang, Xiaoguang Gao, Chris P. Thompson


class TestDBNInference(unittest.TestCase):
    def setUp(self):
        self.dbn = DynamicBayesianNetwork()
        self.dbn.add_edges_from(
            [(('Z', 0), ('X', 0)), (('Z', 0), ('Y', 0)), (('Z', 0), ('Z', 1))])
        cpd_start_z = TabularCPD(('Z', 0), 2, [[0.8, 0.2]])
        cpd_x = TabularCPD(
            ('X', 0), 2, [[0.9, 0.6], [0.1, 0.4]], [('Z', 0)], 2)
        cpd_y = TabularCPD(
            ('Y', 0), 2, [[0.7, 0.2], [0.3, 0.8]], [('Z', 0)], 2)
        cpd_trans_z = TabularCPD(
            ('Z', 1), 2, [[0.9, 0.1], [0.1, 0.9]], [('Z', 0)], 2)
        self.dbn.add_cpds(cpd_start_z, cpd_trans_z, cpd_x, cpd_y)
        self.dbn.initialize_initial_state()
        self.dbn_inference = DBNInference(self.dbn)

    def test_forward_inf_single_variable(self):
        query_result = self.dbn_inference.forward_inference([('X', 0)])
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.84, 0.16]))

    def test_forward_inf_multiple_variable(self):
        query_result = self.dbn_inference.forward_inference(
            [('X', 0), ('Y', 0)])
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.84, 0.16]))
        np_test.assert_array_almost_equal(query_result[('Y', 0)].values,
                                          np.array([0.6, 0.4]))

    def test_forward_inf_single_variable_with_evidence(self):
        query_result = self.dbn_inference.forward_inference([(
            'Z', 1)], {('Y', 0): 0,
                       ('Y', 1): 0})
        np_test.assert_array_almost_equal(query_result[('Z', 1)].values,
                                          np.array([0.95080214, 0.04919786]))

    def test_forward_inf_multiple_variable_with_evidence(self):
        query_result = self.dbn_inference.forward_inference([('Z', 1), (
            'X', 1)], {('Y', 0): 0,
                       ('Y', 1): 0})
        np_test.assert_array_almost_equal(query_result[('Z', 1)].values,
                                          np.array([0.95080214, 0.04919786]))

        np_test.assert_array_almost_equal(query_result[('X', 1)].values,
                                          np.array([0.88524064, 0.11475936]))
