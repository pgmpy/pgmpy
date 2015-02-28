#!/usr/bin/env python3

from pgmpy.base import UndirectedGraph
from pgmpy.tests import help_functions as hf
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
