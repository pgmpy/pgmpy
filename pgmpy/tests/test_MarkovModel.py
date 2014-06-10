import unittest
from pgmpy import MarkovModel as mm
from pgmpy import BayesianModel as bm
from pgmpy import Exceptions
import networkx as nx
import help_functions as hf
import numpy as np


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
    #     self.assertRaises(Exceptions.SelfLoopError, bm.BayesianModel,
    #                       [('a', 'a')])
    #
    # def test_class_init_with_data_cycle(self):
    #     self.assertRaises(Exceptions.CycleError, bm.BayesianModel,
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

    def tearDown(self):
        del self.G


class TestBayesianModelMethods(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel([('a', 'd'), ('b', 'd'),
                                   ('d', 'e'), ('b', 'c')])

    def test_add_states(self):
        self.G.set_states({'a': [1, 2, 3], 'b': [4, 5], 'c': [6, 7]})
        self.assertListEqual(sorted([node['name'] for node in
                                     self.G.node['a']['_states']]), [1, 2, 3])
        self.assertListEqual(self.G.node['a']['_rule_for_states'], [0, 1, 2])
        self.assertFalse(self.G.node['a']['_observed'])
        self.assertListEqual(sorted([node['name'] for node in
                                     self.G.node['b']['_states']]), [4, 5])
        self.assertListEqual(self.G.node['b']['_rule_for_states'], [0, 1])
        self.assertFalse(self.G.node['b']['_observed'])
        self.assertListEqual(sorted([node['name'] for node in
                                     self.G.node['c']['_states']]), [6, 7])
        self.assertListEqual(self.G.node['c']['_rule_for_states'], [0, 1])
        self.assertFalse(self.G.node['c']['_observed'])
        self.G.set_states({'a': [8, 9]})
        self.assertListEqual(sorted([node['name'] for node in
                                     self.G.node['a']['_states']]),
                             [1, 2, 3, 8, 9])
        self.assertListEqual(self.G.node['a']['_rule_for_states'],
                             [0, 1, 2, 3, 4])
        self.assertFalse(self.G.node['a']['_observed'])

    def test_get_states(self):
        self.G = bm.BayesianModel([('a', 'd')])
        self.G.set_states({'a': [1, 2, 3], 'd': [4, 5]})
        self.assertListEqual(list(self.G.get_states('a')), [1, 2, 3])
        self.assertListEqual(list(self.G.get_states('d')), [4, 5])
        self.G.node['a']['_rule_for_states'] = [1, 0, 2]
        self.assertListEqual(list(self.G.get_states('a')), [2, 1, 3])
        self.G.node['d']['_rule_for_states'] = [0, 1]
        self.assertListEqual(list(self.G.get_states('d')), [4, 5])

    def test_update_rule_for_states(self):
        self.G._update_rule_for_states('a', 4)
        self.G._update_rule_for_states('b', 1)
        self.assertListEqual(self.G.node['a']['_rule_for_states'],
                             [0, 1, 2, 3])
        self.assertListEqual(self.G.node['b']['_rule_for_states'], [0])
        self.G._update_rule_for_states('a', 5)
        self.assertListEqual(self.G.node['a']['_rule_for_states'],
                             [0, 1, 2, 3, 4])
        self.G.node['a']['_rule_for_states'] = [1, 0, 2]
        self.G._update_rule_for_states('a', 5)
        self.assertListEqual(self.G.node['a']['_rule_for_states'],
                             [1, 0, 2, 3, 4])

    def test_update_node_observed_status(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.assertFalse(self.G.node['a']['_observed'])
        self.G.node['a']['_states'][0]['observed_status'] = True
        self.G._update_node_observed_status('a')
        self.assertTrue(self.G.node['a']['_observed'])

    def test_no_missing_states(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.assertTrue(self.G._no_missing_states('a', [1, 2, 3]))
        self.assertRaises(ValueError,
                          self.G._no_missing_states, 'a', [1, 2])

    def test_no_extra_states(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.assertTrue(self.G._no_extra_states('a', [1, 2]))
        self.assertTrue(self.G._no_extra_states('a', [1, 2, 3]))
        self.assertRaises(ValueError,
                          self.G._no_extra_states, 'a', [1, 2, 3, 4])

    def test_no_extra_parents(self):
        self.assertTrue(self.G._no_extra_parents('d', ['a', 'b']))
        self.assertRaises(ValueError,
                          self.G._no_extra_parents, 'd', ['a', 'b', 'c'])
        self.assertTrue(self.G._no_extra_parents('d', ['a']))

    def test_no_missing_parents(self):
        self.assertTrue(self.G._no_missing_parents('d', ['a', 'b']))
        self.assertTrue(self.G._no_missing_parents('d', ['a', 'b', 'c']))
        self.assertRaises(ValueError,
                          self.G._no_missing_parents, 'd', ['a'])

    def test_get_rule_for_states(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.assertListEqual(self.G.get_rule_for_states('a'), [1, 2, 3])

    def test_set_rule_for_states(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.G.set_rule_for_states('a', [3, 1, 2])
        self.assertListEqual(self.G.get_rule_for_states('a'), [3, 1, 2])

    def test_all_states_present_in_list(self):
        self.G.set_states({'a': [1, 2, 3]})
        self.assertTrue(self.G._all_states_present_in_list('a', [1, 2, 3]))
        self.assertTrue(self.G._all_states_present_in_list('a', [2, 1, 3]))
        self.assertFalse(self.G._all_states_present_in_list('a', [1, 2]))

    def test_is_node_parents_equal_parents_list(self):
        self.assertTrue(self.G._is_node_parents_equal_parents_list(
            'd', ['a', 'b']))
        self.assertTrue(self.G._is_node_parents_equal_parents_list(
            'd', ['b', 'a']))
        self.assertFalse(self.G._is_node_parents_equal_parents_list(
            'd', ['a']))

    def test_get_rule_for_parents(self):
        self.assertListEqual(self.G.get_rule_for_parents('d'), ['a', 'b'])
        self.assertListEqual(self.G.get_rule_for_parents('a'), [])

    def test_set_rule_for_parents(self):
        self.G.set_rule_for_parents('d', ['b', 'a'])
        self.assertListEqual(self.G.node['d']['_rule_for_parents'], [1, 0])
        self.assertListEqual(self.G.get_rule_for_parents('d'), ['b', 'a'])

    def test_get_parents(self):
        self.assertListEqual(list(self.G.get_parents('d')), ['a', 'b'])
        self.G.set_rule_for_parents('d', ['b', 'a'])
        self.assertListEqual(list(self.G.get_parents('d')), ['b', 'a'])

    def test_get_parent_objects(self):
        self.assertListEqual(list(self.G._get_parent_objects('d')),
                             [self.G.node['a'], self.G.node['b']])
        self.assertListEqual(list(self.G._get_parent_objects('a')), [])

    def test_moral_graph(self):
        moral_graph = self.G.moral_graph()
        self.assertListEqual(sorted(moral_graph.nodes()), ['a', 'b', 'c', 'd', 'e'])
        for edge in moral_graph.edges():
            self.assertTrue(edge in [('a', 'b'), ('a', 'd'), ('b', 'c'), ('d', 'b'), ('e', 'd')] or
                            (edge[1], edge[0]) in [('a', 'b'), ('a', 'd'), ('b', 'c'), ('d', 'b'), ('e', 'd')])

    def test_moral_graph_with_edge_present_over_parents(self):
        G = bm.BayesianModel([('a', 'd'), ('d', 'e'), ('b', 'd'), ('b', 'c'), ('a', 'b')])
        moral_graph = G.moral_graph()
        self.assertListEqual(sorted(moral_graph.nodes()), ['a', 'b', 'c', 'd', 'e'])
        for edge in moral_graph.edges():
            self.assertTrue(edge in [('a', 'b'), ('c', 'b'), ('d', 'a'), ('d', 'b'), ('d', 'e')] or
                            (edge[1], edge[0]) in [('a', 'b'), ('c', 'b'), ('d', 'a'), ('d', 'b'), ('d', 'e')])

    def tearDown(self):
        del self.G


class TestBayesianModelCPD(unittest.TestCase):

    def setUp(self):
        self.G = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
                                   ('i', 's')])
        self.G.add_states({'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'], 'l': ['yes', 'no']})

    # def test_set_cpd(self):
    #     self.G.set_cpd('g', [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])
    #     self.assertIsInstance(self.G.node['g']['_cpd'], bm.CPD.TabularCPD)
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

if __name__ == '__main__':
        unittest.main()

