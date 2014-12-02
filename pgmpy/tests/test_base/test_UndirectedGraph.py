#!/usr/bin/env python3

from pgmpy.base import UndirectedGraph
from pgmpy.factors import Factor
from pgmpy.tests import help_functions as hf
from pgmpy.exceptions import CardinalityError
import numpy as np
import unittest


class TestUndirectedGraphCreation(unittest.TestCase):
    def setUp(self):
        self.graph = UndirectedGraph()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, UndirectedGraph)

    def test_class_init_with_data_string(self):
        self.G = UndirectedGraph([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['a', 'b'], ['b', 'c']])

    def test_add_node_string(self):
        self.graph.add_node('a')
        self.assertListEqual(self.graph.nodes(), ['a'])

    def test_add_node_nonstring(self):
        self.graph.add_node(1)
        self.assertListEqual(self.graph.nodes(), [1])

    def test_add_nodes_from_string(self):
        self.graph.add_nodes_from(['a', 'b', 'c', 'd'])
        self.assertListEqual(sorted(self.graph.nodes()),
                             ['a', 'b', 'c', 'd'])

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

    def test_number_of_neighbors(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertEqual(len(self.graph.neighbors('b')), 2)

    def tearDown(self):
        del self.graph


class TestUndirectedGraphFactorOperations(unittest.TestCase):
    def setUp(self):
        self.graph = UndirectedGraph()

    def test_add_single_factor(self):
        phi = Factor(['a', 'b'], [2, 2], range(4))
        self.graph.add_factors(phi)
        self.assertListEqual(self.graph.get_factors(), [phi])

    def test_add_multiple_factors(self):
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.assertListEqual(self.graph.get_factors(), [phi1, phi2])

    def test_remove_single_factor(self):
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        self.assertListEqual(self.graph.get_factors(), [phi2])

    def test_remove_multiple_factors(self):
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1, phi2)
        self.assertListEqual(self.graph.get_factors(), [])

    def test_partition_function(self):
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertEqual(self.graph.get_partition_function(), 22.0)

    def test_partition_function_raises_error(self):
        phi1 = Factor(['a', 'b'], [2, 2], range(4))
        phi2 = Factor(['b', 'c'], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        self.assertRaises(ValueError,
                          self.graph.get_partition_function)

    def tearDown(self):
        del self.graph


class TestUndirectedGraphTriangulation(unittest.TestCase):
    def setUp(self):
        self.graph = UndirectedGraph()

    def test_check_clique(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])
        self.assertTrue(self.graph.check_clique(['a', 'b', 'c']))

    def test_is_triangulated(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])
        self.assertTrue(self.graph.is_triangulated())

    def test_triangulation_h1_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H1', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h2_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H2', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h3_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H3', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h4_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h5_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h6_inplace(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertTrue(self.graph.is_triangulated())
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_cardinality_mismatch_raises_error(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        factor_list = [Factor(edge, [2, 2], np.random.rand(4)) for edge in
                       self.graph.edges()]
        self.graph.add_factors(*factor_list)
        self.graph.add_factors(Factor(['a', 'b'], [2, 3], np.random.rand(6)))
        self.assertRaises(CardinalityError, self.graph.triangulate)

    def test_triangulation_h1_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H1', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h2_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H2', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'c'], ['a', 'd'],
                              ['b', 'c'], ['c', 'd']])

    def test_triangulation_h3_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H3', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h4_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H4', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h5_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H5', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def test_triangulation_h6_create_new(self):
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'),
                                   ('d', 'a')])
        phi1 = Factor(['a', 'b'], [2, 3], np.random.rand(6))
        phi2 = Factor(['b', 'c'], [3, 4], np.random.rand(12))
        phi3 = Factor(['c', 'd'], [4, 5], np.random.rand(20))
        phi4 = Factor(['d', 'a'], [5, 2], np.random.random(10))
        self.graph.add_factors(phi1, phi2, phi3, phi4)
        H = self.graph.triangulate(heuristic='H6', inplace=True)
        self.assertListEqual(hf.recursive_sorted(H.edges()),
                             [['a', 'b'], ['a', 'd'], ['b', 'c'],
                              ['b', 'd'], ['c', 'd']])

    def tearDown(self):
        del self.graph
