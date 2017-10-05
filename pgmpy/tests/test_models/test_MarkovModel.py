import unittest

import networkx as nx
import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product
from pgmpy.independencies import Independencies
from pgmpy.extern import six
from pgmpy.extern.six.moves import range
from pgmpy.models import BayesianModel, MarkovModel, FactorGraph
from pgmpy.tests import help_functions as hf


class TestMarkovModelCreation(unittest.TestCase):
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
        self.graph = MarkovModel()

    def test_get_cardinality(self):

        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])

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

        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])

        phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(['c', 'd'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertEqual(self.graph.get_cardinality('a'), 2)
        self.assertEqual(self.graph.get_cardinality('b'), 2)
        self.assertEqual(self.graph.get_cardinality('c'), 1)
        self.assertEqual(self.graph.get_cardinality('d'), 2)

    def test_check_model(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])

        phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        self.assertRaises(ValueError, self.graph.check_model)

        phi2 = DiscreteFactor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        self.assertRaises(ValueError, self.graph.check_model)

    def test_check_model1(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(['c', 'b'], [3, 2], np.random.rand(6))
        phi3 = DiscreteFactor(['c', 'd'], [3, 4], np.random.rand(12))
        phi4 = DiscreteFactor(['d', 'a'], [4, 1], np.random.rand(4))

        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.assertTrue(self.graph.check_model())

        self.graph.remove_factors(phi1, phi4)
        phi1 = DiscreteFactor(['a', 'b'], [4, 2], np.random.rand(8))
        self.graph.add_factors(phi1)
        self.assertTrue(self.graph.check_model())

    def test_check_model2(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])

        phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))

        phi2 = DiscreteFactor(['b', 'c'], [3, 3], np.random.rand(9))
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi2)

        phi3 = DiscreteFactor(['c', 'a'], [4, 4], np.random.rand(16))
        self.graph.add_factors(phi3)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi3)

        phi2 = DiscreteFactor(['b', 'c'], [2, 3], np.random.rand(6))
        phi3 = DiscreteFactor(['c', 'd'], [3, 4], np.random.rand(12))
        phi4 = DiscreteFactor(['d', 'a'], [4, 3], np.random.rand(12))
        self.graph.add_factors(phi2, phi3, phi4)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi2, phi3, phi4)

        phi2 = DiscreteFactor(['a', 'b'], [1, 3], np.random.rand(3))
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi2)

    def test_check_model3(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])

        phi1 = DiscreteFactor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi1)

        phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(['a', 'c'], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi1, phi2)

        phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(['b', 'c'], [2, 3], np.random.rand(6))
        phi3 = DiscreteFactor(['c', 'd'], [3, 4], np.random.rand(12))
        phi4 = DiscreteFactor(['d', 'a'], [4, 1], np.random.rand(4))
        phi5 = DiscreteFactor(['d', 'b'], [4, 2], np.random.rand(8))
        self.graph.add_factors(phi1, phi2, phi3, phi4, phi5)
        self.assertRaises(ValueError, self.graph.check_model)
        self.graph.remove_factors(phi1, phi2, phi3, phi4, phi5)

    def test_factor_graph(self):
        phi1 = DiscreteFactor(['Alice', 'Bob'], [3, 2], np.random.rand(6))
        phi2 = DiscreteFactor(['Bob', 'Charles'], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles')])
        self.graph.add_factors(phi1, phi2)

        factor_graph = self.graph.to_factor_graph()
        self.assertIsInstance(factor_graph, FactorGraph)
        self.assertListEqual(sorted(factor_graph.nodes()),
                             ['Alice', 'Bob', 'Charles', 'phi_Alice_Bob',
                              'phi_Bob_Charles'])
        self.assertListEqual(hf.recursive_sorted(factor_graph.edges()),
                             [['Alice', 'phi_Alice_Bob'], ['Bob', 'phi_Alice_Bob'],
                              ['Bob', 'phi_Bob_Charles'], ['Charles', 'phi_Bob_Charles']])
        self.assertListEqual(factor_graph.get_factors(), [phi1, phi2])

    def test_factor_graph_raises_error(self):
        self.graph.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles')])
        self.assertRaises(ValueError, self.graph.to_factor_graph)

    def test_junction_tree(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)

        junction_tree = self.graph.to_junction_tree()
        self.assertListEqual(hf.recursive_sorted(junction_tree.nodes()),
                             [['a', 'b', 'd'], ['b', 'c', 'd']])
        self.assertEqual(len(junction_tree.edges()), 1)

    def test_junction_tree_single_clique(self):

        self.graph.add_edges_from([('x1', 'x2'), ('x2', 'x3'), ('x1', 'x3')])
        phi = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in self.graph.edges()]
        self.graph.add_factors(*phi)

        junction_tree = self.graph.to_junction_tree()
        self.assertListEqual(hf.recursive_sorted(junction_tree.nodes()),
                             [['x1', 'x2', 'x3']])
        factors = junction_tree.get_factors()
        self.assertEqual(factors[0], factor_product(*phi))

    def test_markov_blanket(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.graph.markov_blanket('a'), ['b'])
        self.assertListEqual(sorted(self.graph.markov_blanket('b')),
                             ['a', 'c'])

    def test_local_independencies(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        independencies = self.graph.get_local_independencies()
        self.assertIsInstance(independencies, Independencies)
        self.assertEqual(independencies, Independencies(['a', 'c', 'b']))

    def test_bayesian_model(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)

        bm = self.graph.to_bayesian_model()
        self.assertIsInstance(bm, BayesianModel)
        self.assertListEqual(sorted(bm.nodes()), ['a', 'b', 'c', 'd'])
        self.assertTrue(nx.is_chordal(bm.to_undirected()))

    def tearDown(self):
        del self.graph


class TestUndirectedGraphFactorOperations(unittest.TestCase):
    def setUp(self):
        self.graph = MarkovModel()

    def test_add_factor_raises_error(self):
        self.graph.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles'),
                                   ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        factor = DiscreteFactor(['Alice', 'Bob', 'John'], [2, 2, 2], np.random.rand(8))
        self.assertRaises(ValueError, self.graph.add_factors, factor)

    def test_add_single_factor(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        self.graph.add_factors(phi)
        six.assertCountEqual(self, self.graph.factors, [phi])

    def test_add_multiple_factors(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        six.assertCountEqual(self, self.graph.factors, [phi1, phi2])

    def test_get_factors(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        six.assertCountEqual(self, self.graph.get_factors(), [])
        self.graph.add_factors(phi1, phi2)
        six.assertCountEqual(self, self.graph.get_factors(), [phi1, phi2])
        six.assertCountEqual(self, self.graph.get_factors('a'), [phi1])

    def test_remove_single_factor(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        six.assertCountEqual(self, self.graph.factors, [phi2])

    def test_remove_multiple_factors(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1, phi2)
        six.assertCountEqual(self, self.graph.factors, [])

    def test_partition_function(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertEqual(self.graph.get_partition_function(), 22.0)

    def test_partition_function_raises_error(self):
        self.graph.add_nodes_from(['a', 'b', 'c', 'd'])
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], range(4))
        phi2 = DiscreteFactor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError,
                          self.graph.get_partition_function)

    def tearDown(self):
        del self.graph


class TestUndirectedGraphTriangulation(unittest.TestCase):
    def setUp(self):
        self.graph = MarkovModel()

    def test_check_clique(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])
        self.assertTrue(self.graph.is_clique(['a', 'b', 'c']))

    def test_is_triangulated(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])
        self.assertTrue(self.graph.is_triangulated())

    def test_triangulation_h1_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H1', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h2_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H2', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h3_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H3', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h4_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h5_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h6_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_cardinality_mismatch_raises_error(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        factor_list = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in
                       self.graph.edges()]
        self.graph.add_factors(*factor_list)
        self.graph.add_factors(DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6)))
        self.assertRaises(ValueError, self.graph.triangulate)

    def test_triangulation_h1_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H1', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h2_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H2', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h3_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H3', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h4_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h5_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H5', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h6_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = DiscreteFactor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = DiscreteFactor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = DiscreteFactor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = DiscreteFactor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H6', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_copy(self):
        # Setup the original graph
        self.graph.add_nodes_from(['a', 'b'])
        self.graph.add_edges_from([('a', 'b')])

        # Generate the copy
        copy = self.graph.copy()

        # Ensure the copied model is correct
        self.assertTrue(copy.check_model())

        # Basic sanity checks to ensure the graph was copied correctly
        self.assertEqual(len(copy.nodes()), 2)
        self.assertListEqual(copy.neighbors('a'), ['b'])
        self.assertListEqual(copy.neighbors('b'), ['a'])

        # Modify the original graph ...
        self.graph.add_nodes_from(['c'])
        self.graph.add_edges_from([('c', 'b')])

        # ... and ensure none of those changes get propagated
        self.assertEqual(len(copy.nodes()), 2)
        self.assertListEqual(copy.neighbors('a'), ['b'])
        self.assertListEqual(copy.neighbors('b'), ['a'])
        with self.assertRaises(nx.NetworkXError):
            copy.neighbors('c')

        # Ensure the copy has no factors at this point
        self.assertEqual(len(copy.get_factors()), 0)

        # Add factors to the original graph
        phi1 = DiscreteFactor(['a', 'b'], [2, 2], [[0.3, 0.7], [0.9, 0.1]])
        self.graph.add_factors(phi1)

        # The factors should not get copied over
        with self.assertRaises(AssertionError):
            self.assertListEqual(copy.get_factors(), self.graph.get_factors())

        # Create a fresh copy
        del copy
        copy = self.graph.copy()
        self.assertListEqual(copy.get_factors(), self.graph.get_factors())

        # If we change factors in the original, it should not be passed to the clone
        phi1.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.assertNotEqual(self.graph.get_factors(), copy.get_factors())

        # Start with a fresh copy
        del copy
        self.graph.add_nodes_from(['d'])
        copy = self.graph.copy()

        # Ensure an unconnected node gets copied over as well
        self.assertEqual(len(copy.nodes()), 4)
        self.assertListEqual(self.graph.neighbors('a'), ['b'])
        self.assertTrue('a' in self.graph.neighbors('b'))
        self.assertTrue('c' in self.graph.neighbors('b'))
        self.assertListEqual(self.graph.neighbors('c'), ['b'])
        self.assertListEqual(self.graph.neighbors('d'), [])

        # Verify that changing the copied model should not update the original
        copy.add_nodes_from(['e'])
        self.assertListEqual(copy.neighbors('e'), [])
        with self.assertRaises(nx.NetworkXError):
            self.graph.neighbors('e')

        # Verify that changing edges in the copy doesn't create edges in the original
        copy.add_edges_from([('d', 'b')])

        self.assertTrue('a' in copy.neighbors('b'))
        self.assertTrue('c' in copy.neighbors('b'))
        self.assertTrue('d' in copy.neighbors('b'))

        self.assertTrue('a' in self.graph.neighbors('b'))
        self.assertTrue('c' in self.graph.neighbors('b'))
        self.assertFalse('d' in self.graph.neighbors('b'))

        # If we remove factors from the copied model, it should not reflect in the original
        copy.remove_factors(phi1)
        self.assertEqual(len(self.graph.get_factors()), 1)
        self.assertEqual(len(copy.get_factors()), 0)

    def tearDown(self):
        del self.graph
