#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as np_test
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.factors import TabularCPD
from pgmpy.factors import Factor
from pgmpy.Inference import VariableElimination


class TestVariableElimination(unittest.TestCase):
    def setUp(self):
        self.bayesian = BayesianModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        cpd_a = TabularCPD('a', 2, [[0.4, 0.6]])
        cpd_b = TabularCPD('b', 2, [[0.2, 0.4], [0.8, 0.6]], evidence=['a'], evidence_card=[2])
        cpd_c = TabularCPD('c', 2, [[0.3, 0.4], [0.7, 0.6]], evidence=['b'], evidence_card=[2])
        cpd_d = TabularCPD('d', 2, [[0.7, 0.5], [0.3, 0.5]], evidence=['c'], evidence_card=[2])
        cpd_e = TabularCPD('e', 2, [[0.5, 0.1], [0.5, 0.9]], evidence=['d'], evidence_card=[2])

        self.bayesian.add_cpd([cpd_a, cpd_b, cpd_c, cpd_d, cpd_e])

        self.markov = MarkovModel([('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd')])
        factor_a_b = Factor(['a', 'b'], [2, 2], [100, 40, 40, 100])
        factor_b_c = Factor(['b', 'c'], [2, 2], [60, 20, 20, 40])
        factor_c_d = Factor(['c', 'd'], [2, 2], [80, 50, 50, 60])
        factor_d_a = Factor(['d', 'a'], [2, 2], [10, 90, 40, 20])

        self.markov.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)

    def test_bayesian_ve_1(self):
        model = VariableElimination(self.bayesian)
        phi_e = model.query(variables={'e': {}})
        self.assertEqual(phi_e.variables, ['e'])
        np_test.assert_array_equal(phi_e.cardinality, np.array([2]))
        np_test.assert_array_almost_equal(phi_e.values, np.array([0.32944, 0.67056]))

    def test_bayesian_ve_2(self):
        model = VariableElimination(self.bayesian)
        phi_e_1 = model.query(variables={'e': {1}})
        self.assertEqual(phi_e_1.variables, ['e'])
        np_test.assert_array_equal(phi_e_1.cardinality, [1])
        np_test.assert_array_almost_equal(phi_e_1.values, np.array([0.67056]))

    def test_bayesian_ve_3(self):
        model = VariableElimination(self.bayesian)
        phi_c_d = model.query(variables={'c': {}, 'd': {}})
        self.assertEqual(phi_c_d.variables, ['c', 'd'])
        np_test.assert_array_equal(phi_c_d.cardinality, np.array([2, 2]))
        np_test.assert_array_almost_equal(phi_c_d.values, np.array([0.2576, 0.3160, 0.1104, 0.3160]))

    def test_bayesian_ve_4(self):
        model = VariableElimination(self.bayesian)
        phi_c_given_d = model.query(variables={'c': {}}, conditions={'d': {}})
        self.assertEqual(phi_c_given_d.variables, ['c', 'd'])
        np_test.assert_array_equal(phi_c_given_d.cardinality, np.array([2, 2]))
        np_test.assert_array_almost_equal(phi_c_given_d.values, np.array([0.2245474, 0.2754511,
                                                                          0.12945602, 0.37054548]))

    def test_bayesian_ve_5(self):
        model = VariableElimination(self.bayesian)
        phi_c_given_d_1 = model.query(variables={'c': {}}, conditions={'d': {1}})
        self.assertEqual(phi_c_given_d_1.variables, ['c', 'd'])
        np_test.assert_array_equal(phi_c_given_d_1.cardinality, [2, 1])
        np_test.assert_array_almost_equal(phi_c_given_d_1.values, np.array([0.258911, 0.741088]))

    def test_markov_ve_1(self):
        model = VariableElimination(self.markov)
        phi_a = model.query(variables={'a': {}})
        self.assertEqual(phi_a.variables, ['a'])
        np_test.assert_array_equal(phi_a.cardinality, np.array([2]))
        np_test.assert_array_almost_equal(phi_a.values, np.array([0.59868, 0.40131]))

    def test_markov_ve_2(self):
        model = VariableElimination(self.markov)
        phi_a_1 = model.query(variables={'a': {1}})
        self.assertEqual(phi_a_1.variables, ['a'])
        np_test.assert_array_equal(phi_a_1.cardinality, np.array([1]))
        np_test.assert_array_almost_equal(phi_a_1.values, np.array([0.40131]))

    def test_markov_ve_3(self):
        model = VariableElimination(self.markov)
        phi_c_a = model.query(variables={'c': {}, 'a': {}})
        self.assertEqual(phi_c_a.variables, ['a', 'c'])
        np_test.assert_array_equal(phi_c_a.cardinality, np.array([2, 2]))
        np_test.assert_array_almost_equal(phi_c_a.values, np.array([0.43016194, 0.16852227, 0.22823887, 0.17307692]))

    def test_markov_ve_4(self):
        model = VariableElimination(self.markov)
        phi_c_given_a = model.query(variables={'c': {}}, conditions={'a': {}})
        self.assertEqual(phi_c_given_a.variables, ['a', 'c'])
        np_test.assert_array_equal(phi_c_given_a.cardinality, np.array([2, 2]))
        np_test.assert_array_almost_equal(phi_c_given_a.values, np.array([0.68335, 0.49333, 0.34664, 0.50666]))

    def test_markov_ve_5(self):
        model = VariableElimination(self.markov)
        phi_c_given_a_1 = model.query(variables={'c': {}}, conditions={'a': {1}})
        self.assertEqual(phi_c_given_a_1.variables, ['a', 'c'])
        np_test.assert_array_equal(phi_c_given_a_1.cardinality, np.array([2, 2]))
        np_test.assert_array_almost_equal(phi_c_given_a_1.values, np.array([0.49333, 0.50666]))
