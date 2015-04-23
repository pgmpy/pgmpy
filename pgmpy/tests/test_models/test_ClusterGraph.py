from pgmpy.models import ClusterGraph
from pgmpy.tests import help_functions as hf
from pgmpy.factors import Factor
import unittest
import numpy as np


class TestClusterGraphCreation(unittest.TestCase):
    def setUp(self):
        self.graph = ClusterGraph()

    def test_add_single_node(self):
        self.graph.add_node(('a', 'b'))
        self.assertListEqual(self.graph.nodes(), [('a', 'b')])

    def test_add_single_node_raises_error(self):
        self.assertRaises(TypeError, self.graph.add_node, 'a')

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(hf.recursive_sorted(self.graph.nodes()), [['a', 'b'], ['b', 'c']])

    def test_add_single_edge(self):
        self.graph.add_edge(('a', 'b'), ('b', 'c'))
        self.assertListEqual(hf.recursive_sorted(self.graph.nodes()), [['a', 'b'], ['b', 'c']])
        self.assertListEqual(sorted([node for edge in self.graph.edges() for node in edge]),
                             [('a', 'b'), ('b', 'c')])

    def test_add_single_edge_raises_error(self):
        self.assertRaises(ValueError, self.graph.add_edge, ('a', 'b'), ('c', 'd'))

    def tearDown(self):
        del self.graph


class TestClusterGraphFactorOperations(unittest.TestCase):
    def setUp(self):
        self.graph = ClusterGraph()

    def test_add_single_factor(self):
        self.graph.add_node(('a', 'b'))
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        self.assertListEqual(self.graph.get_factors(), [phi1])

    def test_add_single_factor_raises_error(self):
        self.graph.add_node(('a', 'b'))
        phi1 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.assertRaises(ValueError, self.graph.add_factors, phi1)

    def test_add_multiple_factors(self):
        self.graph.add_edges_from([[('a', 'b'), ('b', 'c')]])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_factors(node=('b', 'a')), phi1)
        self.assertEqual(self.graph.get_factors(node=('b', 'c')), phi2)

    def test_remove_factors(self):
        self.graph.add_edges_from([[('a', 'b'), ('b', 'c')]])
        phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        self.assertListEqual(self.graph.get_factors(), [phi2])

    def test_get_partition_function(self):
        self.graph.add_edges_from([[('a', 'b'), ('b', 'c')]])
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_partition_function(), 22.0)
