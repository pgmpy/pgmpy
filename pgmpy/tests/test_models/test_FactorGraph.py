import numpy as np
import unittest

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
from pgmpy.models import MarkovModel
from pgmpy.models import JunctionTree
from pgmpy.tests import help_functions as hf
from pgmpy.extern import six
from pgmpy.extern.six.moves import range


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

    def tearDown(self):
        del self.graph


class TestFactorGraphFactorOperations(unittest.TestCase):
    def setUp(self):
        self.graph = FactorGraph()

    def test_add_single_factor(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        six.assertCountEqual(self, self.graph.factors, [phi1])

    def test_add_multiple_factors(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        six.assertCountEqual(self, self.graph.factors, [phi1, phi2])

    def test_get_factors(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])

        six.assertCountEqual(self, self.graph.get_factors(), [])

        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_factors(node=phi1), phi1)
        self.assertEqual(self.graph.get_factors(node=phi2), phi2)
        six.assertCountEqual(self, self.graph.get_factors(), [phi1, phi2])

        self.graph.remove_factors(phi1)
        self.assertRaises(ValueError, self.graph.get_factors, node=phi1)

    def test_remove_factors(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        six.assertCountEqual(self, self.graph.factors, [phi2])

    def test_get_partition_function(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_partition_function(), 22.0)

    def tearDown(self):
        del self.graph


class TestFactorGraphMethods(unittest.TestCase):
    def setUp(self):
        self.graph = FactorGraph()

    def test_get_cardinality(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('c', 'phi2'), ('d', 'phi2'),
                                   ('a', 'phi3'), ('d', 'phi3')])

        self.assertDictEqual(self.graph.get_cardinality(), {})

        phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        self.assertDictEqual(self.graph.get_cardinality(), {'a': 1, 'b': 2})
        self.graph.remove_factors(phi1)
        self.assertDictEqual(self.graph.get_cardinality(), {})

        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['c', 'd'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertDictEqual(self.graph.get_cardinality(), {'d': 2, 'a': 2, 'b': 2, 'c': 1})

        phi3 = DiscreteFactor(['d', 'a'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi3)
        self.assertDictEqual(self.graph.get_cardinality(), {'d': 1, 'c': 1, 'b': 2, 'a': 2})

        self.graph.remove_factors(phi1, phi2, phi3)
        self.assertDictEqual(self.graph.get_cardinality(), {})

    def test_get_cardinality_with_node(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_cardinality('a'), 2)
        self.assertEqual(self.graph.get_cardinality('b'), 2)
        self.assertEqual(self.graph.get_cardinality('c'), 2)

    def test_get_factor_nodes(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))

        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        six.assertCountEqual(self, self.graph.get_factor_nodes(), [phi1, phi2])

    def test_get_variable_nodes(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        six.assertCountEqual(self, self.graph.get_variable_nodes(), ['a', 'b', 'c'])

    def test_get_variable_nodes_raises_error(self):
        self.graph.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
                                   ('b', 'phi2'), ('c', 'phi2')])
        self.assertRaises(ValueError, self.graph.get_variable_nodes)

    def test_to_markov_model(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        mm = self.graph.to_markov_model()
        self.assertIsInstance(mm, MarkovModel)
        self.assertListEqual(sorted(mm.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(mm.edges()),
                             [['a', 'b'], ['b', 'c']])
        self.assertListEqual(sorted(mm.get_factors(),
                             key=lambda x: x.scope()), [phi1, phi2])

    def test_to_junction_tree(self):
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])

        self.graph.add_factors(phi1, phi2)
        jt = self.graph.to_junction_tree()
        self.assertIsInstance(jt, JunctionTree)
        self.assertListEqual(hf.recursive_sorted(jt.nodes()),
                             [['a', 'b'], ['b', 'c']])
        self.assertEqual(len(jt.edges()), 1)

    def test_check_model(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        self.assertTrue(self.graph.check_model())

        self.graph.remove_factors(phi1)
        self.graph.remove_node(phi1)
        phi1 = DiscreteFactor(['a', 'b'], [4, 2], np.random.rand(8))
        self.graph.add_factors(phi1)
        self.graph.add_edges_from([('a', phi1)])
        self.assertTrue(self.graph.check_model())

    def test_check_model1(self):
        self.graph.add_nodes_from(['a', 'b', 'c', 'd'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError, self.graph.check_model)

        self.graph.remove_node('d')
        self.assertTrue(self.graph.check_model())

    def test_check_model2(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)

        self.graph.add_edges_from([('a', 'b')])
        self.assertRaises(ValueError, self.graph.check_model)

        self.graph.add_edges_from([(phi1, phi2)])
        self.assertRaises(ValueError, self.graph.check_model)

        self.graph.remove_edges_from([('a', 'b'), (phi1, phi2)])
        self.assertTrue(self.graph.check_model())

    def test_check_model3(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        phi3 = DiscreteFactor(['a', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2, phi3)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi3)
        self.assertTrue(self.graph.check_model())

    def test_check_model4(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [3, 2], np.random.rand(6))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError, self.graph.check_model)

        self.graph.remove_factors(phi2)
        self.graph.remove_node(phi2)
        phi3 = DiscreteFactor(['c', 'a'], [4, 4], np.random.rand(16))
        self.graph.add_factors(phi3)
        self.graph.add_edges_from([('a', phi3), ('c', phi3)])
        self.assertRaises(ValueError, self.graph.check_model)

    def test_copy(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([('a', phi1), ('b', phi1),
                                   ('b', phi2), ('c', phi2)])
        graph_copy = self.graph.copy()
        self.assertIsInstance(graph_copy, FactorGraph)
        self.assertTrue(graph_copy.check_model())
        self.assertEqual(self.graph.get_factors(), graph_copy.get_factors())
        self.graph.remove_factors(phi1, phi2)
        self.assertTrue(phi1 not in self.graph.factors and phi2 not in self.graph.factors)
        self.assertTrue(phi1 in graph_copy.factors and phi2 in graph_copy.factors)
        self.graph.add_factors(phi1, phi2)
        self.graph.factors[0] = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        self.assertNotEqual(self.graph.get_factors()[0], graph_copy.get_factors()[0])
        self.assertNotEqual(self.graph.factors, graph_copy.factors)

    def tearDown(self):
        del self.graph
