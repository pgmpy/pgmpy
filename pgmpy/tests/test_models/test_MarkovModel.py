import unittest
from pgmpy.tests import help_functions as hf
from pgmpy.models import MarkovModel


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.graph = MarkovModel()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, MarkovModel)

    def test_class_init_with_data_string(self):
        self.g = MarkovModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.g.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.g.edges()),
                             [['a', 'b'], ['b', 'c']])

    def test_class_init_with_data_nonstring(self):
        self.g = MarkovModel([(1, 2), (2, 3)])

    def test_add_node_string(self):
        self.graph.add_node('a')
        self.assertListEqual(self.graph.nodes(), ['a'])

    def test_add_node_nonstring(self):
        self.graph.add_node(1)

    def test_add_nodes_from_string(self):
        self.graph.add_nodes_from(['a', 'b', 'c', 'd'])
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'b', 'c', 'd'])

    def test_add_nodes_from_non_string(self):
        self.graph.add_nodes_from([1, 2, 3, 4])

    def test_add_edge_string(self):
        self.graph.add_edge('d', 'e')
        self.assertListEqual(sorted(self.graph.nodes()), ['d', 'e'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['d', 'e']])
        self.graph.add_nodes_from(['a', 'b', 'c'])
        self.graph.add_edge('a', 'b')
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['d', 'e']])

    def test_add_edge_nonstring(self):
        self.graph.add_edge(1, 2)

    def test_add_edge_selfloop(self):
        self.assertRaises(ValueError, self.graph.add_edge, 'a', 'a')

    def test_add_edges_from_string(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['b', 'c']])
        self.graph.add_nodes_from(['d', 'e', 'f'])
        self.graph.add_edges_from([('d', 'e'), ('e', 'f')])
        self.assertListEqual(sorted(self.graph.nodes()),
                             ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             hf.recursive_sorted([('a', 'b'), ('b', 'c'),
                                                  ('d', 'e'), ('e', 'f')]))

    def test_add_edges_from_nonstring(self):
        self.graph.add_edges_from([(1, 2), (2, 3)])

    def test_add_edges_from_self_loop(self):
        self.assertRaises(ValueError, self.graph.add_edges_from,
                          [('a', 'a')])

    def test_number_of_neighbors(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertEqual(len(self.graph.neighbors('b')), 2)

    def tearDown(self):
        del self.graph


class TestMarkovModelMethods(unittest.TestCase):
    def setUp(self):
        self.graph = MarkovModel([('a', 'd'), ('b', 'd'),
                                  ('d', 'e'), ('b', 'c')])

    # def test_set_boolean_states(self):
    #     self.graph.set_boolean_states('d')
    #     self.assertListEqual(sorted(self.graph.get_states('d')), [0, 1])

#     def test_set_add_states(self):
#         self.graph.set_states({'a': [1, 2, 3], 'b': [4, 5]})
#         self.assertListEqual(sorted(self.graph.node['a']['_states']), [1, 2, 3])
#         self.assertFalse(self.graph.node['a']['_is_observed'])
#         self.assertListEqual(sorted(self.graph.node['b']['_states']), [4, 5])
#         self.assertFalse(self.graph.node['b']['_is_observed'])
#         self.graph.add_states({'a': [8, 9]})
#         self.assertListEqual(sorted(self.graph.node['a']['_states']), [1, 2, 3, 8, 9])
#         self.assertFalse(self.graph.node['a']['_is_observed'])
#
#     def test_set_get_states(self):
#         self.graph = mm.MarkovModel([('a', 'd')])
#         self.graph.set_states({'a': [1, 2, 3], 'd': [4, 5]})
#         self.assertListEqual(list(self.graph.get_states('a')), [1, 2, 3])
#         self.assertListEqual(list(self.graph.get_states('d')), [4, 5])
#
#     def test_get_rule_for_states(self):
#         self.graph.set_states({'a': [1, 2, 3]})
#         self.assertListEqual(self.graph.get_rule_for_states('a'), [1, 2, 3])
#
#     def test_set_rule_for_states(self):
#         self.graph.set_states({'a': [1, 2, 3]})
#         self.graph.set_rule_for_states('a', [3, 1, 2])
#         self.assertListEqual(self.graph.get_rule_for_states('a'), [3, 1, 2])
#
#     def test_number_of_states(self):
#         self.graph.set_states({'a': [1, 2, 3]})
#         self.assertEqual(self.graph.number_of_states('a'), 3)
#
#     def test_remove_all_states(self):
#         self.graph.set_states({'a': [1, 2, 3], 'd': [4, 5]})
#         self.assertEqual(sorted(self.graph.get_states('a')), [1, 2, 3])
#         self.graph.remove_all_states('a')
#         self.assertEqual(sorted(self.graph.get_states('a')), [])
#
#     def tearDown(self):
#         del self.graph
#
#
# class TestMarkovModelObservations(unittest.TestCase):
#     def setUp(self):
#         self.graph = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
#                                      ('i', 's')])
#         self.graph.add_states(
#             {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'],
#              'l': ['yes', 'no']})
#
#     # def test_set_cpd(self):
#     #     self.G.set_cpd('g', [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #                          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #                          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])
#     #     self.assertIsInstance(self.G.node['g']['_cpd'], mm.CPD.TabularCPD)
#     #     np.testing.assert_array_equal(self.G.node['g']['_cpd'].cpd, np.array((
#     #         [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])))
#     #
#     # def test_get_cpd(self):
#     #     self.G.set_cpd('g', [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #                          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #                          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])
#     #     np.testing.assert_array_equal(self.G.get_cpd('g'), np.array((
#     #         [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #          [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     #          [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]])))
#
#     def test_is_observed(self):
#         self.graph.set_observations({'d': 'easy'})
#         self.assertTrue(self.graph.is_observed('d'))
#         self.assertFalse(self.graph.is_observed('i'))
#
#     def test_set_observations_single_state(self):
#         self.graph.set_observations({'d': 'easy'})
#         self.assertTrue(self.graph.is_observed('d'))
#         self.assertEqual(self.graph.get_observation('d'), 'easy')
#
#     def test_set_observation_multiple_state(self):
#         self.graph.set_observations({'d': 'easy', 'g': 'A'})
#         self.assertTrue(self.graph.is_observed('d'))
#         self.assertEqual(self.graph.get_observation('d'), 'easy')
#         self.assertTrue(self.graph.is_observed('g'))
#         self.assertEqual(self.graph.get_observation('g'), 'A')
#
#     def test_set_observation_multiple_state_not_found_observation(self):
#         self.assertRaises(ValueError, self.graph.set_observations, {'d': 'unknow_state'})
#
#     def test_unset_observations_single_state(self):
#         self.graph.set_observations({'d': 'easy', 'g': 'A'})
#         self.assertEqual(self.graph.get_observation('d'), 'easy')
#         self.assertTrue(self.graph.is_observed('d'))
#         self.graph.unset_observation('d')
#         self.assertFalse(self.graph.is_observed('d'))
#         self.assertRaises(ValueError, self.graph.get_observation, 'd')
#
#     def test_unset_observations_multiple_state(self):
#         self.graph.set_observations({'d': 'easy', 'g': 'A', 'i': 'dumb'})
#         self.graph.unset_observations(['d', 'i'])
#         self.assertTrue(self.graph.is_observed('g'))
#         self.assertEqual(self.graph.get_observation('g'), 'A')
#         self.assertFalse(self.graph.is_observed('d'))
#         self.assertRaises(ValueError, self.graph.get_observation, 'd')
#         self.assertFalse(self.graph.is_observed('i'))
#         self.assertRaises(ValueError, self.graph.get_observation, 'i')
#
#     def test_reset_observations_node_error(self):
#         self.assertRaises(KeyError, self.graph.unset_observation, 'j')
#
#     def tearDown(self):
#         del self.graph
#
#
# class TestMarkovModelFactors(unittest.TestCase):
#     def setUp(self):
#         self.graph = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
#                                      ('i', 's')])
#         self.graph.add_states(
#             {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'],
#              'l': ['yes', 'no']})
#
#     def test_add_factors(self):
#         factor = self.graph.add_factor(['d', 'g'], range(6))
#         self.assertListEqual(self.graph.get_factors(), [factor])
#         self.assertListEqual(sorted(factor.get_variables()), ['d', 'g'])
#         self.assertEqual(factor.get_value([0, 0]), 0)
#         self.assertEqual(factor.get_value([0, 1]), 2)
#
#     def test_invalid_nodes(self):
#         self.assertRaises(ValueError, self.graph.add_factor, ['d', 'i'], range(4))
#
#     def test_invalid_potentials(self):
#         self.assertRaises(ValueError, self.graph.add_factor, ['d', 'g'], range(5))
#
#     def test_normalization_constant_brute_force(self):
#         self.graph.add_factor(['d', 'g'], range(6))
#         self.graph.add_factor(['i', 'g'], range(6))
#         self.graph.add_factor(['g', 'l'], [0.1] * 6)
#         self.graph.add_factor(['i', 's'], [0.1] * 6)
#         z = self.graph.normalization_constant_brute_force()
#         self.assertAlmostEqual(z, 6.42, places=2)
#
#     def test_make_jt(self):
#         self.graph.add_factor(['d', 'g'], range(6))
#         self.graph.add_factor(['i', 'g'], range(6))
#         self.graph.add_factor(['g', 'l'], [0.1] * 6)
#         self.graph.add_factor(['i', 's'], [0.1] * 6)
#         factors = set(self.graph.get_factors())
#         jt = self.graph.make_jt(2)
#         jtfactors = set([jt.node[jtnode]["factor"] for jtnode in jt.nodes()])
#         # This works in this case, as none of the factors get multiplied in this case.
#         #be careful with other cases
#         self.assertEqual(factors, jtfactors)
#
#     def tearDown(self):
#         del self.graph

if __name__ == '__main__':
    unittest.main()
