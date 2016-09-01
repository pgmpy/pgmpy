import unittest
import numpy as np
import numpy.testing as np_test

from pgmpy.inference import DBNInference
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# The sample Dynamic Bayesian Network is taken from the following paper:-
# Novel recursive inference algorithm for discrete dynamic Bayesian networks
# Huange Wang, Xiaoguang Gao, Chris P. Thompson


class TestDBNInference(unittest.TestCase):

    def setUp(self):
        dbn_1 = DynamicBayesianNetwork()
        dbn_1.add_edges_from(
            [(('Z', 0), ('X', 0)), (('Z', 0), ('Y', 0)), (('Z', 0), ('Z', 1))])
        cpd_start_z_1 = TabularCPD(('Z', 0), 2, [[0.8, 0.2]])
        cpd_x_1 = TabularCPD(
            ('X', 0), 2, [[0.9, 0.6], [0.1, 0.4]], [('Z', 0)], [2])
        cpd_y_1 = TabularCPD(
            ('Y', 0), 2, [[0.7, 0.2], [0.3, 0.8]], [('Z', 0)], [2])
        cpd_trans_z_1 = TabularCPD(
            ('Z', 1), 2, [[0.9, 0.1], [0.1, 0.9]], [('Z', 0)], [2])
        dbn_1.add_cpds(cpd_start_z_1, cpd_trans_z_1, cpd_x_1, cpd_y_1)
        dbn_1.initialize_initial_state()
        self.dbn_inference_1 = DBNInference(dbn_1)
        dbn_2 = DynamicBayesianNetwork()
        dbn_2.add_edges_from([(('Z', 0), ('X', 0)), (('X', 0), ('Y', 0)),
                              (('Z', 0), ('Z', 1))])
        cpd_start_z_2 = TabularCPD(('Z', 0), 2, [[0.5, 0.5]])
        cpd_x_2 = TabularCPD(
            ('X', 0), 2, [[0.6, 0.9], [0.4, 0.1]], [('Z', 0)], [2])
        cpd_y_2 = TabularCPD(
            ('Y', 0), 2, [[0.2, 0.3], [0.8, 0.7]], [('X', 0)], [2])
        cpd_z_2 = TabularCPD(
            ('Z', 1), 2, [[0.4, 0.7], [0.6, 0.3]], [('Z', 0)], [2])
        dbn_2.add_cpds(cpd_x_2, cpd_y_2, cpd_z_2, cpd_start_z_2)
        dbn_2.initialize_initial_state()
        self.dbn_inference_2 = DBNInference(dbn_2)

    def test_forward_inf_single_variable(self):
        query_result = self.dbn_inference_1.forward_inference([('X', 0)])
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.84, 0.16]))

    def test_forward_inf_multiple_variable(self):
        query_result = self.dbn_inference_1.forward_inference(
            [('X', 0), ('Y', 0)])
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.84, 0.16]))
        np_test.assert_array_almost_equal(query_result[('Y', 0)].values,
                                          np.array([0.6, 0.4]))

    def test_forward_inf_single_variable_with_evidence(self):
        query_result = self.dbn_inference_1.forward_inference([(
            'Z', 1)], {('Y', 0): 0,
                       ('Y', 1): 0})
        np_test.assert_array_almost_equal(query_result[('Z', 1)].values,
                                          np.array([0.95080214, 0.04919786]))
        query_result = self.dbn_inference_2.forward_inference([(
            'X', 2)], {('Y', 0): 1,
                       ('Y', 1): 0,
                       ('Y', 2): 1})
        np_test.assert_array_almost_equal(query_result[('X', 2)].values,
                                          np.array([0.76738736, 0.23261264]))

    def test_forward_inf_multiple_variable_with_evidence(self):
        query_result = self.dbn_inference_1.forward_inference([('Z', 1), (
            'X', 1)], {('Y', 0): 0,
                       ('Y', 1): 0})
        np_test.assert_array_almost_equal(query_result[('Z', 1)].values,
                                          np.array([0.95080214, 0.04919786]))

        np_test.assert_array_almost_equal(query_result[('X', 1)].values,
                                          np.array([0.88524064, 0.11475936]))

    def test_backward_inf_single_variable(self):
        query_result = self.dbn_inference_2.backward_inference([('Y', 0)])
        np_test.assert_array_almost_equal(query_result[('Y', 0)].values,
                                          np.array([0.225, 0.775]))

    def test_backward_inf_multiple_variables(self):
        query_result = self.dbn_inference_2.backward_inference([('X', 0),
                                                                ('Y', 0)])
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.75, 0.25]))
        np_test.assert_array_almost_equal(query_result[('Y', 0)].values,
                                          np.array([0.225, 0.775]))

    def test_backward_inf_single_variable_with_evidence(self):
        query_result = self.dbn_inference_2.backward_inference([(
            'X', 0)], {('Y', 0): 0,
                       ('Y', 1): 1,
                       ('Y', 2): 1})
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.66594382, 0.33405618]))

        query_result = self.dbn_inference_1.backward_inference([(
            'Z', 1)], {('Y', 0): 0,
                       ('Y', 1): 0,
                       ('Y', 2): 0})
        np_test.assert_array_almost_equal(query_result[('Z', 1)].values,
                                          np.array([0.98048698, 0.01951302]))

    def test_backward_inf_multiple_variables_with_evidence(self):
        query_result = self.dbn_inference_2.backward_inference([('X', 0), (
            'X', 1)], {('Y', 0): 0,
                       ('Y', 1): 1,
                       ('Y', 2): 1})
        np_test.assert_array_almost_equal(query_result[('X', 0)].values,
                                          np.array([0.66594382, 0.33405618]))
        np_test.assert_array_almost_equal(query_result[('X', 1)].values,
                                          np.array([0.7621772, 0.2378228]))
