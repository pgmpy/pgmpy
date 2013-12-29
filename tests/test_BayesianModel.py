import unittest
import BayesianModel.BayesianModel as bm
import networkx as nx
import help_functions as hf


class TestModel(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = bm.BayesianModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.g.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.g.edges()),
                             [['a', 'b'], ['b', 'c']])

    def test_class_init_with_data_nonstring(self):
        self.assertRaises(TypeError, bm.BayesianModel, [(1, 2), (2, 3)])

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
        self.assertListEqual(self.G.edges(), [('d', 'e')])
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edge('a', 'b')
        self.assertListEqual(hf.recursive_sorted(self.G.edges()), [['a', 'b'], ['d', 'e']])

    def test_add_edge_nonstring(self):
        self.assertRaises(TypeError, self.G.add_edge, 1, 2)

    def test_add_edges_from_string(self):
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()), [['a', 'b'], ['b', 'c']])
        self.G.add_nodes_from(['d', 'e', 'f'])
        self.G.add_edges_from([('d', 'e'), ('e', 'f')])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()), hf.recursive_sorted([('a', 'b'), ('b', 'c'), ('d', 'e'), ('e', 'f')]))

    def test_add_edges_from_nonstring(self):
        self.assertRaises(TypeError, self.G.add_edges_from, [(1, 2), (2, 3)])

    def test_update_node_parents_bm_constructor(self):
        self.g = bm.BayesianModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.g.predecessors('a'), [])
        self.assertListEqual(self.g.predecessors('b'), ['a'])
        self.assertListEqual(self.g.predecessors('c'), ['b'])

    def test_update_node_parents(self):
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.G.predecessors('a'), [])
        self.assertListEqual(self.G.predecessors('b'), ['a'])
        self.assertListEqual(self.G.predecessors('c'), ['b'])


    # def test_add_edges(self):
    #     self.G.add_node('a', 'b', 'c', 'd')
    #     self.G.add_edges([('a', 'b'), ('c', 'd')])
    #     self.assertListEqual(hf.recursive_sorted(self.G.edges()),
    #                          [('a', 'b'), ('c', 'd')])
    #
    # def test_add_edges_both_tuples(self):
    #     self.G.add_nodes('a', 'b', 'c', 'd')
    #     self.G.add_edges([('a', 'b'), ('c', 'd')])
    #     self.assertListEqual(hf.recursive_sorted(self.G.edges()),
    #                          [('a', 'b'), ('c', 'd')])
    #
    # def test_add_edges_tail_string(self):
    #     self.G.add_nodes('a', 'b', 'c', 'd')
    #     self.G.add_edges(('a', 'b'), ('a', 'c'))
    #     self.assertListEqual(hf.recursive_sorted(self.G.edges()),
    #                          hf.recursive_sorted([('a', 'b'), ('a', 'c')]))
    #
    # def test_add_edges_head_string(self):
    #     self.G.add_nodes('a', 'b', 'c', 'd')
    #     self.G.add_edges(('a', 'c'), ('a', 'b'))
    #     self.assertListEqual(hf.recursive_sorted(self.G.edges()),
    #                          hf.recursive_sorted([('a', 'c'), ('b', 'c')]))
    #
    # def test_add_edges_both_string(self):
    #     self.G.add_nodes('a', 'b')
    #     self.G.add_edges(('a', 'b'))
    #     self.assertListEqual(hf.recursive_sorted(self.G.edges()),
    #                          hf.recursive_sorted([('a', 'b')]))
    #
    # def test_add_edges_multiple_times(self):
    #     self.G.add_nodes('a', 'b', 'c', 'd')
    #     self.G.add_edges(('a', 'c'), ('a', 'd'))
    #     self.G.add_edges(('b', 'c'), ('b', 'd'))
    #     self.assertListEqual(hf.recursive_sorted(self.G.edges()),
    #                          hf.recursive_sorted([('a', 'c'), ('a', 'd'),
    #                                              ('b', 'c'), ('b', 'd')]))

    def tearDown(self):
        del self.G


# class TestNodeProperties(unittest.TestCase):
#
#     def setUp(self):
#         self.G = bm.BayesianModel()
#         self.G.add_nodes('a', 'b', 'c', 'd', 'e')
#         self.G.add_edges(('a', 'b'), ('c', 'd'))
#
#     # def test_parents(self):
#     #     self.assertListEqual(sorted(self.G.node['c']['_parents']),
#     #                          sorted(['a', 'b']))
#     #     self.assertListEqual(sorted(self.G.node['d']['_parents']),
#     #                          sorted(['a', 'b']))
# # TODO       self.assertRaises(KeyError, self.G.node['a']['_parents'])
# # TODO       self.assertRaises(KeyError, self.G.node['b']['_parents'])
#
# # TODO add test_default_rule
#
# # TODO check test_add_states again
#     def test_add_states(self):
#         self.G.add_states('a', ('test_state_3', 'test_state_1',
#                                 'test_state_2'))
#         self.G.add_states('b', ('test_state_2', 'test_state_1'))
#         self.G.add_states('c', ('test_state_1',))
#         self.G.add_states('d', ('test_state_1', 'test_state_2'))
#         self.assertEqual(self.G.node['a']['_states'],
#                          [{'name': 'test_state_3', 'observed_status': False},
#                           {'name': 'test_state_1', 'observed_status': False},
#                           {'name': 'test_state_2', 'observed_status': False}])
#         self.assertEqual(self.G.node['b']['_states'],
#                          [{'name': 'test_state_2', 'observed_status': False},
#                           {'name': 'test_state_1', 'observed_status': False}])
#         self.assertEqual(self.G.node['c']['_states'],
#                          [{'name': 'test_state_1', 'observed_status': False}])
#         self.assertEqual(self.G.node['d']['_states'],
#                          [{'name': 'test_state_1', 'observed_status': False},
#                           {'name': 'test_state_2', 'observed_status': False}])
#
# # TODO add test_default_rule_for_states
#
# # TODO    def test_rule_for_states
# # TODO check test_states_fuction again
#     def test_states_function(self):
#         self.G.add_states('a', ('test_state_3', 'test_state_1',
#                                 'test_state_2'))
#         self.G.add_states('b', ('test_state_2', 'test_state_1'))
#         self.G.add_states('c', ('test_state_1',))
#         self.G.add_states('d', ('test_state_1', 'test_state_2'))
#         states = {'a': [], 'b': [], 'c': [], 'd': []}
#         nodes = ['a', 'b', 'c', 'd']
#         for node in nodes:
#             for state in self.G.get_states(node):
#                 states[node].append(state)
#         self.assertEqual(states['a'], ['test_state_3', 'test_state_1',
#                                        'test_state_2'])
#         self.assertEqual(states['b'], ['test_state_2', 'test_state_1'])
#         self.assertEqual(states['c'], ['test_state_1'])
#         self.assertEqual(states['d'], ['test_state_1', 'test_state_2'])
#
#     def tearDown(self):
#         del self.G
#

if __name__ == '__main__':
        unittest.main()
