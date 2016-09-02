import unittest
import numpy as np
import numpy.testing as np_test

from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.models import JunctionTree
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.extern.six.moves import range


class TestVariableElimination(unittest.TestCase):
    def setUp(self):
        self.bayesian_model = BayesianModel([('A', 'J'), ('R', 'J'), ('J', 'Q'),
                                             ('J', 'L'), ('G', 'L')])
        cpd_a = TabularCPD('A', 2, values=[[0.2], [0.8]])
        cpd_r = TabularCPD('R', 2, values=[[0.4], [0.6]])
        cpd_j = TabularCPD('J', 2, values=[[0.9, 0.6, 0.7, 0.1],
                                           [0.1, 0.4, 0.3, 0.9]],
                           evidence=['A', 'R'], evidence_card=[2, 2])
        cpd_q = TabularCPD('Q', 2, values=[[0.9, 0.2], [0.1, 0.8]],
                           evidence=['J'], evidence_card=[2])
        cpd_l = TabularCPD('L', 2, values=[[0.9, 0.45, 0.8, 0.1],
                                           [0.1, 0.55, 0.2, 0.9]],
                           evidence=['J', 'G'], evidence_card=[2, 2])
        cpd_g = TabularCPD('G', 2, values=[[0.6], [0.4]])
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)

        self.bayesian_inference = VariableElimination(self.bayesian_model)

    # All the values that are used for comparision in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        query_result = self.bayesian_inference.query(['J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))

    def test_query_multiple_variable(self):
        query_result = self.bayesian_inference.query(['Q', 'J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.4912, 0.5088]))

    def test_query_single_variable_with_evidence(self):
        query_result = self.bayesian_inference.query(variables=['J'],
                                                     evidence={'A': 0, 'R': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.60, 0.40]))

    def test_query_multiple_variable_with_evidence(self):
        query_result = self.bayesian_inference.query(variables=['J', 'Q'],
                                                     evidence={'A': 0, 'R': 0,
                                                               'G': 0, 'L': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.818182, 0.181818]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.772727, 0.227273]))

    def test_query_multiple_times(self):
        # This just tests that the models are not getting modified while querying them
        query_result = self.bayesian_inference.query(['J'])
        query_result = self.bayesian_inference.query(['J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))

        query_result = self.bayesian_inference.query(['Q', 'J'])
        query_result = self.bayesian_inference.query(['Q', 'J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.4912, 0.5088]))

        query_result = self.bayesian_inference.query(variables=['J'],
                                                     evidence={'A': 0, 'R': 1})
        query_result = self.bayesian_inference.query(variables=['J'],
                                                     evidence={'A': 0, 'R': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.60, 0.40]))

        query_result = self.bayesian_inference.query(variables=['J', 'Q'],
                                                     evidence={'A': 0, 'R': 0,
                                                               'G': 0, 'L': 1})
        query_result = self.bayesian_inference.query(variables=['J', 'Q'],
                                                     evidence={'A': 0, 'R': 0,
                                                               'G': 0, 'L': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.818182, 0.181818]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.772727, 0.227273]))

    def test_max_marginal(self):
        np_test.assert_almost_equal(self.bayesian_inference.max_marginal(), 0.1659, decimal=4)

    def test_max_marginal_var(self):
        np_test.assert_almost_equal(self.bayesian_inference.max_marginal(['G']), 0.5714, decimal=4)

    def test_max_marginal_var1(self):
        np_test.assert_almost_equal(self.bayesian_inference.max_marginal(['G', 'R']),
                                    0.4055, decimal=4)

    def test_max_marginal_var2(self):
        np_test.assert_almost_equal(self.bayesian_inference.max_marginal(['G', 'R', 'A']),
                                    0.3260, decimal=4)

    def test_map_query(self):
        map_query = self.bayesian_inference.map_query()
        self.assertDictEqual(map_query, {'A': 1, 'R': 1, 'J': 1, 'Q': 1, 'G': 0,
                                         'L': 0})

    def test_map_query_with_evidence(self):
        map_query = self.bayesian_inference.map_query(['A', 'R', 'L'],
                                                      {'J': 0, 'Q': 1, 'G': 0})
        self.assertDictEqual(map_query, {'A': 1, 'R': 0, 'L': 0})

    def test_induced_graph(self):
        induced_graph = self.bayesian_inference.induced_graph(['G', 'Q', 'A', 'J', 'L', 'R'])
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        self.assertEqual([['A', 'J'], ['A', 'R'], ['G', 'J'], ['G', 'L'],
                          ['J', 'L'], ['J', 'Q'], ['J', 'R'], ['L', 'R']],
                         result_edges)

    def test_induced_width(self):
        result_width = self.bayesian_inference.induced_width(['G', 'Q', 'A', 'J', 'L', 'R'])
        self.assertEqual(2, result_width)

    def tearDown(self):
        del self.bayesian_inference
        del self.bayesian_model


class TestVariableEliminationMarkov(unittest.TestCase):
    def setUp(self):
        # It is just a moralised version of the above Bayesian network so all the results are same. Only factors
        # are under consideration for inference so this should be fine.
        self.markov_model = MarkovModel([('A', 'J'), ('R', 'J'), ('J', 'Q'), ('J', 'L'),
                                         ('G', 'L'), ('A', 'R'), ('J', 'G')])

        factor_a = TabularCPD('A', 2, values=[[0.2], [0.8]]).to_factor()
        factor_r = TabularCPD('R', 2, values=[[0.4], [0.6]]).to_factor()
        factor_j = TabularCPD('J', 2, values=[[0.9, 0.6, 0.7, 0.1],
                                              [0.1, 0.4, 0.3, 0.9]],
                              evidence=['A', 'R'], evidence_card=[2, 2]).to_factor()
        factor_q = TabularCPD('Q', 2, values=[[0.9, 0.2], [0.1, 0.8]],
                              evidence=['J'], evidence_card=[2]).to_factor()
        factor_l = TabularCPD('L', 2, values=[[0.9, 0.45, 0.8, 0.1],
                                              [0.1, 0.55, 0.2, 0.9]],
                              evidence=['J', 'G'], evidence_card=[2, 2]).to_factor()
        factor_g = TabularCPD('G', 2, [[0.6], [0.4]]).to_factor()

        self.markov_model.add_factors(factor_a, factor_r, factor_j, factor_q, factor_l, factor_g)
        self.markov_inference = VariableElimination(self.markov_model)

    # All the values that are used for comparision in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        query_result = self.markov_inference.query(['J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))

    def test_query_multiple_variable(self):
        query_result = self.markov_inference.query(['Q', 'J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.4912, 0.5088]))

    def test_query_single_variable_with_evidence(self):
        query_result = self.markov_inference.query(variables=['J'],
                                                   evidence={'A': 0, 'R': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.60, 0.40]))

    def test_query_multiple_variable_with_evidence(self):
        query_result = self.markov_inference.query(variables=['J', 'Q'],
                                                   evidence={'A': 0, 'R': 0,
                                                             'G': 0, 'L': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.818182, 0.181818]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.772727, 0.227273]))

    def test_query_multiple_times(self):
        # This just tests that the models are not getting modified while querying them
        query_result = self.markov_inference.query(['J'])
        query_result = self.markov_inference.query(['J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))

        query_result = self.markov_inference.query(['Q', 'J'])
        query_result = self.markov_inference.query(['Q', 'J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.4912, 0.5088]))

        query_result = self.markov_inference.query(variables=['J'],
                                                   evidence={'A': 0, 'R': 1})
        query_result = self.markov_inference.query(variables=['J'],
                                                   evidence={'A': 0, 'R': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.60, 0.40]))

        query_result = self.markov_inference.query(variables=['J', 'Q'],
                                                   evidence={'A': 0, 'R': 0,
                                                             'G': 0, 'L': 1})
        query_result = self.markov_inference.query(variables=['J', 'Q'],
                                                   evidence={'A': 0, 'R': 0,
                                                             'G': 0, 'L': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.818182, 0.181818]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.772727, 0.227273]))

    def test_max_marginal(self):
        np_test.assert_almost_equal(self.markov_inference.max_marginal(), 0.1659, decimal=4)

    def test_max_marginal_var(self):
        np_test.assert_almost_equal(self.markov_inference.max_marginal(['G']), 0.5714, decimal=4)

    def test_max_marginal_var1(self):
        np_test.assert_almost_equal(self.markov_inference.max_marginal(['G', 'R']),
                                    0.4055, decimal=4)

    def test_max_marginal_var2(self):
        np_test.assert_almost_equal(self.markov_inference.max_marginal(['G', 'R', 'A']),
                                    0.3260, decimal=4)

    def test_map_query(self):
        map_query = self.markov_inference.map_query()
        self.assertDictEqual(map_query, {'A': 1, 'R': 1, 'J': 1, 'Q': 1, 'G': 0, 'L': 0})

    def test_map_query_with_evidence(self):
        map_query = self.markov_inference.map_query(['A', 'R', 'L'],
                                                    {'J': 0, 'Q': 1, 'G': 0})
        self.assertDictEqual(map_query, {'A': 1, 'R': 0, 'L': 0})

    def test_induced_graph(self):
        induced_graph = self.markov_inference.induced_graph(['G', 'Q', 'A', 'J', 'L', 'R'])
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        self.assertEqual([['A', 'J'], ['A', 'R'], ['G', 'J'], ['G', 'L'],
                          ['J', 'L'], ['J', 'Q'], ['J', 'R'], ['L', 'R']],
                         result_edges)

    def test_induced_width(self):
        result_width = self.markov_inference.induced_width(['G', 'Q', 'A', 'J', 'L', 'R'])
        self.assertEqual(2, result_width)

    def tearDown(self):
        del self.markov_inference
        del self.markov_model


class TestBeliefPropagation(unittest.TestCase):
    def setUp(self):
        self.junction_tree = JunctionTree([(('A', 'B'), ('B', 'C')),
                                           (('B', 'C'), ('C', 'D'))])
        phi1 = DiscreteFactor(['A', 'B'], [2, 3], range(6))
        phi2 = DiscreteFactor(['B', 'C'], [3, 2], range(6))
        phi3 = DiscreteFactor(['C', 'D'], [2, 2], range(4))
        self.junction_tree.add_factors(phi1, phi2, phi3)

        self.bayesian_model = BayesianModel([('A', 'J'), ('R', 'J'), ('J', 'Q'),
                                             ('J', 'L'), ('G', 'L')])
        cpd_a = TabularCPD('A', 2, values=[[0.2], [0.8]])
        cpd_r = TabularCPD('R', 2, values=[[0.4], [0.6]])
        cpd_j = TabularCPD('J', 2, values=[[0.9, 0.6, 0.7, 0.1],
                                           [0.1, 0.4, 0.3, 0.9]],
                           evidence=['A', 'R'], evidence_card=[2, 2])
        cpd_q = TabularCPD('Q', 2, values=[[0.9, 0.2], [0.1, 0.8]],
                           evidence=['J'], evidence_card=[2])
        cpd_l = TabularCPD('L', 2, values=[[0.9, 0.45, 0.8, 0.1],
                                           [0.1, 0.55, 0.2, 0.9]],
                           evidence=['J', 'G'], evidence_card=[2, 2])
        cpd_g = TabularCPD('G', 2, values=[[0.6], [0.4]])
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)

    def test_calibrate_clique_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
        belief_propagation.calibrate()
        clique_belief = belief_propagation.get_clique_beliefs()

        phi1 = DiscreteFactor(['A', 'B'], [2, 3], range(6))
        phi2 = DiscreteFactor(['B', 'C'], [3, 2], range(6))
        phi3 = DiscreteFactor(['C', 'D'], [2, 2], range(4))

        b_A_B = phi1 * (phi3.marginalize(['D'], inplace=False) * phi2).marginalize(['C'], inplace=False)
        b_B_C = phi2 * (phi1.marginalize(['A'], inplace=False) * phi3.marginalize(['D'], inplace=False))
        b_C_D = phi3 * (phi1.marginalize(['A'], inplace=False) * phi2).marginalize(['B'], inplace=False)

        np_test.assert_array_almost_equal(clique_belief[('A', 'B')].values, b_A_B.values)
        np_test.assert_array_almost_equal(clique_belief[('B', 'C')].values, b_B_C.values)
        np_test.assert_array_almost_equal(clique_belief[('C', 'D')].values, b_C_D.values)

    def test_calibrate_sepset_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
        belief_propagation.calibrate()
        sepset_belief = belief_propagation.get_sepset_beliefs()

        phi1 = DiscreteFactor(['A', 'B'], [2, 3], range(6))
        phi2 = DiscreteFactor(['B', 'C'], [3, 2], range(6))
        phi3 = DiscreteFactor(['C', 'D'], [2, 2], range(4))

        b_B = (phi1 * (phi3.marginalize(['D'], inplace=False) *
                       phi2).marginalize(['C'], inplace=False)).marginalize(['A'], inplace=False)

        b_C = (phi2 * (phi1.marginalize(['A'], inplace=False) *
                       phi3.marginalize(['D'], inplace=False))).marginalize(['B'], inplace=False)

        np_test.assert_array_almost_equal(sepset_belief[frozenset((('A', 'B'), ('B', 'C')))].values, b_B.values)
        np_test.assert_array_almost_equal(sepset_belief[frozenset((('B', 'C'), ('C', 'D')))].values, b_C.values)

    def test_max_calibrate_clique_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
        belief_propagation.max_calibrate()
        clique_belief = belief_propagation.get_clique_beliefs()

        phi1 = DiscreteFactor(['A', 'B'], [2, 3], range(6))
        phi2 = DiscreteFactor(['B', 'C'], [3, 2], range(6))
        phi3 = DiscreteFactor(['C', 'D'], [2, 2], range(4))

        b_A_B = phi1 * (phi3.maximize(['D'], inplace=False) * phi2).maximize(['C'], inplace=False)
        b_B_C = phi2 * (phi1.maximize(['A'], inplace=False) * phi3.maximize(['D'], inplace=False))
        b_C_D = phi3 * (phi1.maximize(['A'], inplace=False) * phi2).maximize(['B'], inplace=False)

        np_test.assert_array_almost_equal(clique_belief[('A', 'B')].values, b_A_B.values)
        np_test.assert_array_almost_equal(clique_belief[('B', 'C')].values, b_B_C.values)
        np_test.assert_array_almost_equal(clique_belief[('C', 'D')].values, b_C_D.values)

    def test_max_calibrate_sepset_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
        belief_propagation.max_calibrate()
        sepset_belief = belief_propagation.get_sepset_beliefs()

        phi1 = DiscreteFactor(['A', 'B'], [2, 3], range(6))
        phi2 = DiscreteFactor(['B', 'C'], [3, 2], range(6))
        phi3 = DiscreteFactor(['C', 'D'], [2, 2], range(4))

        b_B = (phi1 * (phi3.maximize(['D'], inplace=False) *
                       phi2).maximize(['C'], inplace=False)).maximize(['A'], inplace=False)

        b_C = (phi2 * (phi1.maximize(['A'], inplace=False) *
                       phi3.maximize(['D'], inplace=False))).maximize(['B'], inplace=False)

        np_test.assert_array_almost_equal(sepset_belief[frozenset((('A', 'B'), ('B', 'C')))].values, b_B.values)
        np_test.assert_array_almost_equal(sepset_belief[frozenset((('B', 'C'), ('C', 'D')))].values, b_C.values)

    # All the values that are used for comparision in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(['J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))

    def test_query_multiple_variable(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(['Q', 'J'])
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.416, 0.584]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.4912, 0.5088]))

    def test_query_single_variable_with_evidence(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(variables=['J'],
                                                evidence={'A': 0, 'R': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.60, 0.40]))

    def test_query_multiple_variable_with_evidence(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(variables=['J', 'Q'],
                                                evidence={'A': 0, 'R': 0,
                                                          'G': 0, 'L': 1})
        np_test.assert_array_almost_equal(query_result['J'].values,
                                          np.array([0.818182, 0.181818]))
        np_test.assert_array_almost_equal(query_result['Q'].values,
                                          np.array([0.772727, 0.227273]))

    def test_map_query(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        map_query = belief_propagation.map_query()
        self.assertDictEqual(map_query, {'A': 1, 'R': 1, 'J': 1, 'Q': 1, 'G': 0,
                                         'L': 0})

    def test_map_query_with_evidence(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        map_query = belief_propagation.map_query(['A', 'R', 'L'],
                                                 {'J': 0, 'Q': 1, 'G': 0})
        self.assertDictEqual(map_query, {'A': 1, 'R': 0, 'L': 0})

    def tearDown(self):
        del self.junction_tree
        del self.bayesian_model
