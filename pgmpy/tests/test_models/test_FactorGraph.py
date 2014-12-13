from pgmpy.factors import Factor
from pgmpy.models import FactorGraph
from pgmpy.models import MarkovModel
from pgmpy.models import JunctionTree
from pgmpy.tests import help_functions as hf
import numpy as np
import unittest


class TestFactorGraphCreation(unittest.TestCase):
    def setUp(self):
        self.graph = FactorGraph()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, FactorGraph)

    def test_class_init_data_string(self):
        self.graph = FactorGraph([('a', 'phi1'), ('b', 'phi1')])
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'b', 'phi1'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'phi1'], ['b', 'phi1']])

    def test_add_single_node(self):
        self.graph.add_node('phi1')
        self.assertEqual(self.graph.nodes(), ['phi1'])

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from(['a', 'b', 'phi1'])
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'b', 'phi1'])

    def test_add_single_edge(self):
        self.graph.add_edge('a', 'phi1')
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'phi1'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'phi1']])

    def test_add_multiple_edges(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1')])
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'b', 'phi1'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'phi1'], ['b', 'phi1']])

    def test_add_self_loop_raises_error(self):
        self.assertRaises(ValueError, self.graph.add_edge, 'a', 'a')

    def test_add_edge_between_variable_nodes_raises_error(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1')])
        self.assertRaises(ValueError, self.graph.add_edge, 'a', 'b')

    def tearDown(self):
        del self.graph


class TestFactorGraphFactorOperations(unittest.TestCase):
    def setUp(self):
        self.graph = FactorGraph()

    def test_add_single_factor(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        self.assertListEqual(self.graph.get_factors(), [phi1])

    def test_add_multiple_factors(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_factors(node='phi1'), phi1)
        self.assertEqual(self.graph.get_factors(node='phi2'), phi2)

    def test_remove_factors(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        self.assertListEqual(self.graph.get_factors(), [phi2])

    def test_get_partition_function(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_partition_function(), 22.0)

    def tearDown(self):
        del self.graph


class TestFactorGraphMethods(unittest.TestCase):
    def setUp(self):
        self.graph = FactorGraph()

    def test_get_factor_nodes(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.assertListEqual(sorted(self.graph.get_factor_nodes()),
                             ['phi1', 'phi2'])

    def test_get_variable_nodes(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.assertListEqual(sorted(self.graph.get_variable_nodes()),
                             ['a', 'b', 'c'])

    def test_get_variable_nodes_raises_error(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        self.assertRaises(ValueError, self.graph.get_variable_nodes)

    def test_to_markov_model(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        mm = self.graph.to_markov_model()
        self.assertIsInstance(mm, MarkovModel)
        self.assertListEqual(sorted(mm.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(mm.edges()),
                             [['a', 'b'], ['b', 'c']])
        self.assertListEqual(sorted(mm.get_factors(),
                             key=lambda x: x.scope()), [phi1, phi2])

    def test_to_junction_tree(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        jt = self.graph.to_junction_tree()
        self.assertIsInstance(jt, JunctionTree)
        self.assertListEqual(hf.recursive_sorted(jt.nodes()),
                             [['a', 'b'], ['b', 'c']])
        self.assertEqual(len(jt.edges()), 1)

    def tearDown(self):
        del self.graph
