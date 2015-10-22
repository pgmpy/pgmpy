import unittest
import numpy as np

from pgmpy.models import ClusterGraph
from pgmpy.tests import help_functions as hf
from pgmpy.factors import Factor
from pgmpy.extern.six.moves import range


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


class TestClusterGraphMethods(unittest.TestCase):
    def setUp(self):
        self.graph = ClusterGraph()

    def test_get_cardinality(self):

        self.graph.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
                                   (('a', 'b', 'c'), ('a', 'c'))])

        self.assertDictEqual(self.graph.get_cardinality(), {})

        phi1 = Factor(['a', 'b', 'c'], [1, 2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        self.assertDictEqual(self.graph.get_cardinality(), {'a': 1, 'b': 2, 'c':2})
        self.graph.remove_factors(phi1)
        self.assertDictEqual(self.graph.get_cardinality(), {})

        phi1 = Factor(['a', 'b'], [1, 2], np.random.rand(2))
        phi2 = Factor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertDictEqual(self.graph.get_cardinality(), {'a': 1, 'b': 2, 'c': 2})

        phi3 = Factor(['a', 'c'], [1, 1], np.random.rand(1))
        self.graph.add_factors(phi3)
        self.assertDictEqual(self.graph.get_cardinality(), {'c': 1, 'b': 2, 'a': 1})

        self.graph.remove_factors(phi1, phi2, phi3)
        self.assertDictEqual(self.graph.get_cardinality(), {})

    def test_check_model(self):
        self.graph.add_edges_from([(('a', 'b'), ('a', 'c'))])
        phi1 = Factor(['a', 'b'], [1, 2], np.random.rand(2))
        phi2 = Factor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertTrue(self.graph.check_model())

        self.graph.remove_factors(phi2)
        phi2 = Factor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        self.assertTrue(self.graph.check_model())

    def test_check_model1(self):
        self.graph.add_edges_from([(('a', 'b'), ('a', 'c')), 
                                   (('a', 'c'), ('a', 'd'))])
        phi1 = Factor(['a', 'b'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        self.assertRaises(ValueError, self.graph.check_model)
        phi2 = Factor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        self.assertRaises(ValueError, self.graph.check_model)

    def test_check_model2(self):
        self.graph.add_edges_from([(('a', 'b'), ('a', 'c')), 
                                   (('a', 'c'), ('a', 'd'))])

        phi1 = Factor(['a', 'b'], [1, 2], np.random.rand(2))
        phi2 = Factor(['a', 'c'], [3, 3], np.random.rand(9))
        phi3 = Factor(['a', 'd'], [4, 4], np.random.rand(16))
        self.graph.add_factors(phi1, phi2, phi3)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi2)
        
        phi2 = Factor(['a', 'c'], [1, 3], np.random.rand(3))
        self.graph.add_factors(phi2)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi3)

        phi3 = Factor(['a', 'd'], [1, 4], np.random.rand(4))
        self.graph.add_factors(phi3)
        self.assertTrue(self.graph.check_model())

    def tearDown(self):
        del self.graph
