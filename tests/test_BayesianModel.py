import unittest
import BayesianModel.BayesianModel as bm
import networkx as nx
import help_functions as hf


class TestModel(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel()

    def test_class_init(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_add_nodes_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.assertListEqual(sorted(self.G.nodes()),
                             sorted(['a', 'b', 'c', 'd']))

    def test_add_nodes_non_string(self):
        self.assertRaises(TypeError, self.G.add_nodes, [1, 2, 3, 4])

    def test_add_edges_both_tuples(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges(('a', 'b'), ('c', 'd'))
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'c'), ('a', 'd'),
                                                 ('b', 'c'), ('b', 'd')]))

    def test_add_edges_tail_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges('a', ('b', 'c'))
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'b'), ('a', 'c')]))

    def test_add_edges_head_string(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges(('a', 'b'), 'c')
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'c'), ('b', 'c')]))

    def test_add_edges_both_string(self):
        self.G.add_nodes('a', 'b')
        self.G.add_edges('a', 'b')
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'b')]))

    def test_add_edges_multiple_times(self):
        self.G.add_nodes('a', 'b', 'c', 'd')
        self.G.add_edges('a', ('c', 'd'))
        self.G.add_edges('b', ('c', 'd'))
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'c'), ('a', 'd'),
                                                 ('b', 'c'), ('b', 'd')]))

    def tearDown(self):
        del self.G


class TestNodeProperties(unittest.TestCase):

    def setUp(self):
        self.G = bm.BayesianModel()
        self.G.add_nodes('a', 'b', 'c', 'd', 'e')
        self.G.add_edges(('a', 'b'), ('c', 'd'))

    def test_parents(self):
        self.assertListEqual(sorted(self.G.node['c']['_parents']),
                             sorted(['a', 'b']))
        self.assertListEqual(sorted(self.G.node['d']['_parents']),
                             sorted(['a', 'b']))
# TODO       self.assertRaises(KeyError, self.G.node['a']['_parents'])
# TODO       self.assertRaises(KeyError, self.G.node['b']['_parents'])

# TODO add test_default_rule

# TODO check test_add_states again
    def test_add_states(self):
        self.G.add_states('a', ('test_state_3', 'test_state_1',
                                'test_state_2'))
        self.G.add_states('b', ('test_state_2', 'test_state_1'))
        self.G.add_states('c', ('test_state_1',))
        self.G.add_states('d', ('test_state_1', 'test_state_2'))
        self.assertEqual(self.G.node['a']['_states'],
                         [{'name': 'test_state_3', 'observed_status': False},
                          {'name': 'test_state_1', 'observed_status': False},
                          {'name': 'test_state_2', 'observed_status': False}])
        self.assertEqual(self.G.node['b']['_states'],
                         [{'name': 'test_state_2', 'observed_status': False},
                          {'name': 'test_state_1', 'observed_status': False}])
        self.assertEqual(self.G.node['c']['_states'],
                         [{'name': 'test_state_1', 'observed_status': False}])
        self.assertEqual(self.G.node['d']['_states'],
                         [{'name': 'test_state_1', 'observed_status': False},
                          {'name': 'test_state_2', 'observed_status': False}])

# TODO add test_default_rule_for_states

# TODO    def test_rule_for_states
# TODO check test_states_fuction again
    def test_states_function(self):
        self.G.add_states('a', ('test_state_3', 'test_state_1',
                                'test_state_2'))
        self.G.add_states('b', ('test_state_2', 'test_state_1'))
        self.G.add_states('c', ('test_state_1',))
        self.G.add_states('d', ('test_state_1', 'test_state_2'))
        states = {'a': [], 'b': [], 'c': [], 'd': []}
        nodes = ['a', 'b', 'c', 'd']
        for node in nodes:
            for state in self.G.get_states(node):
                states[node].append(state)
        self.assertEqual(states['a'], ['test_state_3', 'test_state_1',
                                       'test_state_2'])
        self.assertEqual(states['b'], ['test_state_2', 'test_state_1'])
        self.assertEqual(states['c'], ['test_state_1'])
        self.assertEqual(states['d'], ['test_state_1', 'test_state_2'])

    def tearDown(self):
        del self.G


if __name__ == '__main__':
        unittest.main()
