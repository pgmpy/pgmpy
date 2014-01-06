import unittest
import BayesianModel.BayesianModel as bm
from Exceptions import Exceptions
import networkx as nx
import help_functions as hf


class TestBaseModelCreation(unittest.TestCase):

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

    def test_class_init_with_data_selfloop(self):
        self.assertRaises(Exceptions.SelfLoopError, bm.BayesianModel,
                          [('a', 'a')])

    def test_class_init_with_data_cycle(self):
        self.assertRaises(Exceptions.CycleError, bm.BayesianModel, [('a', 'b'), ('b', 'c'), ('c', 'a')])

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
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['a', 'b'], ['d', 'e']])

    def test_add_edge_nonstring(self):
        self.assertRaises(TypeError, self.G.add_edge, 1, 2)

    def test_add_edge_selfloop(self):
        self.assertRaises(Exceptions.SelfLoopError, self.G.add_edge, 'a', 'a')

    def test_add_edge_result_cycle(self):
        self.G.add_edges_from([('a', 'b'), ('a', 'c')])
        self.assertRaises(Exceptions.CycleError, self.G.add_edge, 'c', 'a')

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
        self.assertRaises(Exceptions.SelfLoopError, self.G.add_edges_from,
                          [('a', 'a')])

    def test_add_edges_from_result_cycle(self):
        self.assertRaises(Exceptions.CycleError, self.G.add_edges_from, [('a', 'b'), ('b', 'c'), ('c', 'a')])

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

    def test_update_rule_for_parents(self):
        self.G.add_edges_from([('a', 'c'), ('b', 'c'), ('a', 'd')])
        self.assertListEqual(self.G.node['a']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['b']['_rule_for_parents'], [0])
        self.assertListEqual(self.G.node['c']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['d']['_rule_for_parents'], [])
        self.G.add_edge('c', 'e')
        self.assertListEqual(self.G.node['a']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['b']['_rule_for_parents'], [0])
        self.assertListEqual(self.G.node['c']['_rule_for_parents'], [0])
        self.assertListEqual(self.G.node['d']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['e']['_rule_for_parents'], [])
        self.G.add_edges_from([('b', 'f'), ('c', 'f')])
        self.assertListEqual(self.G.node['a']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['b']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['c']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['d']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['e']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['f']['_rule_for_parents'], [])
        self.G.add_node('g')
        self.assertListEqual(self.G.node['a']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['b']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['c']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['d']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['e']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['f']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['g']['_rule_for_parents'], [])
        self.G.add_nodes_from(['h', 'i'])
        self.assertListEqual(self.G.node['a']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['b']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['c']['_rule_for_parents'], [0, 1])
        self.assertListEqual(self.G.node['d']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['e']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['f']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['g']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['h']['_rule_for_parents'], [])
        self.assertListEqual(self.G.node['i']['_rule_for_parents'], [])

    def tearDown(self):
        del self.G


class TestBayesianModelMethods(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel([('a', 'd'), ('b', 'd'), ('d', 'e'), ('b', 'c')])

    def test_add_states(self):
        self.G.add_states('a', [1, 2, 3])
        self.G.add_states('b', [4, 5])
        self.G.add_states('c', [6, 7])
        self.assertListEqual(sorted([node['name'] for node in self.G.node['a']['_states']]), [1, 2, 3])
        self.assertListEqual(self.G.node['a']['_rule_for_states'], [0, 1, 2])
        self.assertFalse(self.G.node['a']['_observed'])
        self.assertListEqual(sorted([node['name'] for node in self.G.node['b']['_states']]), [4, 5])
        self.assertListEqual(self.G.node['b']['_rule_for_states'], [0, 1])
        self.assertFalse(self.G.node['b']['_observed'])
        self.assertListEqual(sorted([node['name'] for node in self.G.node['c']['_states']]), [6, 7])
        self.assertListEqual(self.G.node['c']['_rule_for_states'], [0, 1])
        self.assertFalse(self.G.node['c']['_observed'])
        self.G.add_states('a', [8, 9])
        self.assertListEqual(sorted([node['name'] for node in self.G.node['a']['_states']]), [1, 2, 3, 8, 9])
        self.assertListEqual(self.G.node['a']['_rule_for_states'], [0, 1, 2, 3, 4])
        self.assertFalse(self.G.node['a']['_observed'])

    def test_get_states(self):
        self.G = bm.BayesianModel([('a', 'd')])
        self.G.add_states('a', [1, 2, 3])
        self.G.add_states('d', [4, 5])
        self.assertListEqual(self.G.get_states('a'), [1, 2, 3])
        self.assertListEqual(self.G.get_states('d'), [4, 5])
        self.G.node['a']['_rule_for_states'] = [1, 0, 2]
        self.assertListEqual(self.G.get_states('a'), [2, 1, 3])
        self.G.node['d']['_rule_for_states'] = [0, 1]
        self.assertListEqual(self.G.get_states('d'), [4, 5])

    def test_update_rule_for_states(self):
        self.G._update_rule_for_states('a', 4)
        self.G._update_rule_for_states('b', 1)
        self.assertListEqual(self.G.node['a']['_rule_for_states'], [0, 1, 2, 3])
        self.assertListEqual(self.G.node['b']['_rule_for_states'], [0])
        self.G._update_rule_for_states('a', 5)
        self.assertListEqual(self.G.node['a']['_rule_for_states'], [0, 1, 2, 3, 4])
        self.G.node['a']['_rule_for_states'] = [1, 0, 2]
        self.G._update_rule_for_states('a', 5)
        self.assertListEqual(self.G.node['a']['_rule_for_states'], [1, 0, 2, 3, 4])

    def test_update_node_observed_status(self):
        self.G.add_states('a', [1, 2, 3])
        self.assertFalse(self.G.node['a']['_observed'])
        self.G.node['a']['_states'][0]['observed_status'] = True
        self.G._update_node_observed_status('a')
        self.assertTrue(self.G.node['a']['_observed'])

    def test_no_missing_states(self):
        self.G.add_states('a', [1, 2, 3])
        self.assertTrue(self.G._no_missing_states('a', [1, 2, 3]))
        self.assertRaises(Exceptions.MissingStatesError, self.G._no_missing_states, 'a', [1, 2])

    def test_no_extra_states(self):
        self.G.add_states('a', [1, 2, 3])
        self.assertTrue(self.G._no_extra_states('a', [1, 2]))
        self.assertTrue(self.G._no_extra_states('a', [1, 2, 3]))
        self.assertRaises(Exceptions.ExtraStatesError, self.G._no_extra_states, 'a', [1, 2, 3, 4])

    def test_no_extra_parents(self):
        self.assertTrue(self.G._no_extra_parents('d', ['a', 'b']))
        self.assertRaises(Exceptions.ExtraParentsError, self.G._no_extra_parents, 'd', ['a', 'b', 'c'])
        self.assertTrue(self.G._no_extra_parents('d', ['a']))

    def test_no_missing_parents(self):
        self.assertTrue(self.G._no_missing_parents('d', ['a', 'b']))
        self.assertTrue(self.G._no_missing_parents('d', ['a', 'b', 'c']))
        self.assertRaises(Exceptions.MissingParentsError, self.G._no_missing_parents, 'd', ['a'])

    def tearDown(self):
        del self.G


if __name__ == '__main__':
        unittest.main()
