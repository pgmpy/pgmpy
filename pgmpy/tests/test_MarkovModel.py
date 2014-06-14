
import unittest
import numpy as np
import networkx as nx
from pgmpy import MarkovModel as mm
import help_functions as hf




class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.G = mm.MarkovModel()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, mm.UndirectedGraph)

    def test_class_init_with_data_string(self):
        self.g = mm.MarkovModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.g.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.g.edges()),
                             [['a', 'b'], ['b', 'c']])

    def test_class_init_with_data_nonstring(self):
        self.assertRaises(TypeError, mm.MarkovModel, [(1, 2), (2, 3)])

    #TODO: Correct these tests
    # def test_class_init_with_data_selfloop(self):
    #     self.assertRaises(Exceptions.SelfLoopError, mm.MarkovModel,
    #                       [('a', 'a')])
    #
    # def test_class_init_with_data_cycle(self):
    #     self.assertRaises(Exceptions.CycleError, mm.MarkovModel,
    #                       [('a', 'b'), ('b', 'c'), ('c', 'a')])

    def test_add_node_string(self):
        self.G.add_node('a')
        self.assertListEqual(self.G.nodes(), ['a'])

    def test_add_node_nonstring(self):
        self.assertRaises(TypeError, self.G.add_node, 1)

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(['a', 'b', 'c', 'd'])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c', 'd'])

    def test_add_nodes_from_non_string(self):
        self.assertRaises(TypeError, self.G.add_nodes_from, [1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge('d', 'e')
        self.assertListEqual(sorted(self.G.nodes()), ['d', 'e'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['d', 'e']])
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edge('a', 'b')
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['a', 'b'], ['d', 'e']])

    def test_add_edge_nonstring(self):
        self.assertRaises(TypeError, self.G.add_edge, 1, 2)

    def test_add_edge_selfloop(self):
        self.assertRaises(ValueError, self.G.add_edge, 'a', 'a')


    def test_add_edges_from_string(self):
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['a', 'b'], ['b', 'c']])
        self.G.add_nodes_from(['d', 'e', 'f'])
        self.G.add_edges_from([('d', 'e'), ('e', 'f')])
        self.assertListEqual(sorted(self.G.nodes()),
                             ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'b'), ('b', 'c'),
                                                  ('d', 'e'), ('e', 'f')]))

    def test_add_edges_from_nonstring(self):
        self.assertRaises(TypeError, self.G.add_edges_from, [(1, 2), (2, 3)])

    def test_add_edges_from_self_loop(self):
        self.assertRaises(ValueError, self.G.add_edges_from,
                          [('a', 'a')])

    def test_number_of_neighbors(self):
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertEqual(self.G.number_of_neighbours('b'),2)

    def tearDown(self):
        del self.G


class TestMarkovModelMethods(unittest.TestCase):

    def setUp(self):
        self.G = mm.MarkovModel([('a', 'd'), ('b', 'd'),
                                   ('d', 'e'), ('b', 'c')])

    def test_set_boolean_states(self):
        self.G.set_boolean_states('d')
        self.assertListEqual(sorted(self.G.get_states('d')),[0,1])

    def test_set_add_states(self):
        self.G.set_states({'a': [1, 2, 3], 'b': [4, 5]})
        self.assertListEqual(sorted(self.G.node['a']['_states']), [1, 2, 3])
        self.assertFalse(self.G.node['a']['_is_observed'])
        self.assertListEqual(sorted(self.G.node['b']['_states']), [4, 5])
        self.assertFalse(self.G.node['b']['_is_observed'])
        self.G.add_states({'a': [8, 9]})
        self.assertListEqual(sorted(self.G.node['a']['_states']), [1, 2, 3, 8, 9])
        self.assertFalse(self.G.node['a']['_is_observed'])

    def test_set_get_states(self):
        self.G = mm.MarkovModel([('a', 'd')])
        self.G.set_states({'a': [1, 2, 3], 'd': [4, 5]})
        self.assertListEqual(list(self.G.get_states('a')), [1, 2, 3])
        self.assertListEqual(list(self.G.get_states('d')), [4, 5])

    def test_get_rule_for_states(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.assertListEqual(self.G.get_rule_for_states('a'), [1, 2, 3])

    def test_set_rule_for_states(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.G.set_rule_for_states('a', [3, 1, 2])
        self.assertListEqual(self.G.get_rule_for_states('a'), [3, 1, 2])

    def test_number_of_states(self):
        self.G.set_states({'a':[1,2,3]})
        self.assertEqual(self.G.number_of_states('a'),3)

    def test_remove_all_states(self):
        self.G.set_states({'a':[1,2,3], 'd': [4, 5]})
        self.assertEqual(sorted(self.G.get_states('a')), [1,2,3])
        self.G.remove_all_states('a')
        self.assertEqual(sorted(self.G.get_states('a')), [])

    def tearDown(self):
        del self.G


class TestMarkovModelObservations(unittest.TestCase):

    def setUp(self):
        self.G = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
                                   ('i', 's')])
        self.G.add_states({'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'], 'l': ['yes', 'no']})

    # def test_set_cpd(self):
    #     self.G.set_cpd('g', [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])
    #     self.assertIsInstance(self.G.node['g']['_cpd'], mm.CPD.TabularCPD)
    #     np.testing.assert_array_equal(self.G.node['g']['_cpd'].cpd, np.array((
    #         [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])))
    #
    # def test_get_cpd(self):
    #     self.G.set_cpd('g', [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])
    #     np.testing.assert_array_equal(self.G.get_cpd('g'), np.array((
    #         [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])))

    def test_is_observed(self):
        self.G.set_observations({'d': 'easy'})
        self.assertTrue(self.G.is_observed('d'))
        self.assertFalse(self.G.is_observed('i'))


    def test_set_observations_single_state(self):
        self.G.set_observations({'d': 'easy'})
        self.assertTrue(self.G.is_observed('d'))
        self.assertEqual(self.G.get_observation('d'),'easy')

    def test_set_observation_multiple_state(self):
        self.G.set_observations({'d': 'easy', 'g': 'A'})
        self.assertTrue(self.G.is_observed('d'))
        self.assertEqual(self.G.get_observation('d'),'easy')
        self.assertTrue(self.G.is_observed('g'))
        self.assertEqual(self.G.get_observation('g'),'A')

    def test_set_observation_multiple_state_not_found_observation(self):
        self.assertRaises(ValueError, self.G.set_observations, {'d': 'unknow_state'})

    def test_unset_observations_single_state(self):
        self.G.set_observations({'d': 'easy', 'g': 'A'})
        self.assertEqual(self.G.get_observation('d'),'easy')
        self.assertTrue(self.G.is_observed('d'))
        self.G.unset_observation('d')
        self.assertFalse(self.G.is_observed('d'))
        self.assertEqual(self.G.get_observation('d'),'')

    def test_unset_observations_multiple_state(self):
        self.G.set_observations({'d': 'easy', 'g': 'A', 'i': 'dumb'})
        self.G.unset_observations(['d','i'])
        self.assertTrue(self.G.is_observed('g'))
        self.assertEqual(self.G.get_observation('g'),'A')
        self.assertFalse(self.G.is_observed('d'))
        self.assertEqual(self.G.get_observation('d'),'')
        self.assertFalse(self.G.is_observed('i'))
        self.assertEqual(self.G.get_observation('i'),'')

    def test_reset_observations_node_error(self):
        self.assertRaises(KeyError, self.G.unset_observation, 'j')


    # def test_get_ancestros_observation(self):
    #     self.G.set_observations({'d': 'easy', 'g': 'A'})
    #     self.assertListEqual(list(self.G._get_ancestors_observation(['d'])), [])
    #     self.assertListEqual(list(sorted(self.G._get_ancestors_observation(['d', 'g']))), ['d', 'i'])

    def test_get_observed_list(self):
        self.G.set_observations({'d': 'hard', 'i': 'smart'})
        self.assertListEqual(sorted(self.G._get_observed_list()), ['d', 'i'])

    def tearDown(self):
        del self.G

class TestMarkovModelFactors(unittest.TestCase):
    def setUp(self):
        self.G = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
                                   ('i', 's')])
        self.G.add_states({'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'], 'l': ['yes', 'no']})

    def test_add_factors(self):
        factor = self.G.add_factor(['d','g'], range(6))
        self.assertListEqual(self.G.get_factors(),[factor])
        self.assertListEqual(sorted(factor.get_variables()), ['d','g'])
        self.assertEqual(factor.get_value([0,0]),0)
        self.assertEqual(factor.get_value([0,1]),2)

    def test_invalid_nodes(self):
        self.assertRaises(ValueError, self.G.add_factor, ['d','i'], range(4))

    def test_invalid_potentials(self):
        self.assertRaises(ValueError, self.G.add_factor, ['d','g'], range(5))

    def test_normalization_constant_brute_force(self):
        factor = self.G.add_factor(['d','g'],range(6))
        factor = self.G.add_factor(['i','g'],range(6))
        factor = self.G.add_factor(['g','l'],[0.1]*6)
        factor = self.G.add_factor(['i','s'],[0.1]*6)
        Z = self.G.normalization_constant_brute_force()
        self.assertAlmostEqual(Z, 6.42, places=2)

    def test_make_jt(self):
        self.G.add_factor(['d','g'],range(6))
        self.G.add_factor(['i','g'],range(6))
        self.G.add_factor(['g','l'],[0.1]*6)
        self.G.add_factor(['i','s'],[0.1]*6)
        factors =set(self.G.get_factors())
        jt = self.G.make_jt(2)
        jtfactors = set([factor for jtnode in jt.nodes()
                         for factor in jt.node[jtnode]["factors"] ])
        self.assertEqual(factors, jtfactors)

    def tearDown(self):
        del self.G

if __name__ == '__main__':
        unittest.main()

