import unittest
import networkx as nx
import pandas as pd
import numpy as np
import numpy.testing as np_test
from pgmpy.models import NaiveBayes
import pgmpy.tests.help_functions as hf
from pgmpy.factors import TabularCPD
from pgmpy.independencies import Independencies
from pgmpy.extern import six

class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.G = NaiveBayes()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = NaiveBayes([('a', 'b'), ('a', 'c')])
        six.assertCountEqual(self, self.g.nodes(), ['a', 'b', 'c'])
        six.assertCountEqual(self, self.g.edges(), [('a', 'b'), ('a', 'c')])
        self.assertEqual(self.g.parent_node, 'a')
        six.assertCountEqual(self, self.g.children_nodes, ['b', 'c'])

        self.assertRaises(ValueError, NaiveBayes, [('a', 'b'), ('b', 'c')])
        self.assertRaises(ValueError, NaiveBayes, [('a', 'b'), ('c', 'b')])
        self.assertRaises(ValueError, NaiveBayes, [('a', 'b'), ('d', 'e')])

    def test_class_init_with_data_nonstring(self):
        self.g = NaiveBayes([(1, 2), (1, 3)])
        six.assertCountEqual(self, self.g.nodes(), [1, 2, 3])
        six.assertCountEqual(self, self.g.edges(), [(1, 2), (1, 3)])
        self.assertEqual(self.g.parent_node, 1)
        six.assertCountEqual(self, self.g.children_nodes, [2, 3])

        self.assertRaises(ValueError, NaiveBayes, [(1, 2), (2, 3)])
        self.assertRaises(ValueError, NaiveBayes, [(1, 2), (3, 2)])
        self.assertRaises(ValueError, NaiveBayes, [(1, 2), (3, 4)])

    def test_add_node_string(self):
        self.G.add_node('a')
        self.assertListEqual(self.G.nodes(), ['a'])

    def test_add_node_nonstring(self):
        self.G.add_node(1)
        self.assertListEqual(self.G.nodes(), [1])

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(['a', 'b', 'c', 'd'])
        six.assertCountEqual(self, self.G.nodes(), ['a', 'b', 'c', 'd'])

    def test_add_nodes_from_non_string(self):
        self.G.add_nodes_from([1, 2, 3, 4])
        six.assertCountEqual(self, self.G.nodes(), [1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge('a', 'b')
        six.assertCountEqual(self, self.G.nodes(), ['a', 'b'])
        self.assertListEqual(self.G.edges(), [('a', 'b')])
        self.assertEqual(self.G.parent_node, 'a')
        six.assertCountEqual(self, self.G.children_nodes, ['b'])

        self.G.add_nodes_from(['c', 'd'])
        self.G.add_edge('a', 'c')
        self.G.add_edge('a', 'd')
        six.assertCountEqual(self, self.G.nodes(), ['a', 'b', 'c', 'd'])
        six.assertCountEqual(self, self.G.edges(), [('a', 'b'), ('a', 'c'), ('a', 'd')])
        self.assertEqual(self.G.parent_node, 'a')
        six.assertCountEqual(self, self.G.children_nodes, ['b', 'c', 'd'])

        self.assertRaises(ValueError, self.G.add_edge, 'b', 'c')
        self.assertRaises(ValueError, self.G.add_edge, 'd', 'f')
        self.assertRaises(ValueError, self.G.add_edge, 'e', 'f')
        self.assertRaises(ValueError, self.G.add_edges_from, [('a', 'e'), ('b', 'f')])
        self.assertRaises(ValueError, self.G.add_edges_from, [('b', 'f')])

    def test_add_edge_nonstring(self):
        self.G.add_edge(1, 2)
        six.assertCountEqual(self, self.G.nodes(), [1, 2])
        self.assertListEqual(self.G.edges(), [(1, 2)])
        self.assertEqual(self.G.parent_node, 1)
        six.assertCountEqual(self, self.G.children_nodes, [2])

        self.G.add_nodes_from([3, 4])
        self.G.add_edge(1, 3)
        self.G.add_edge(1, 4)
        six.assertCountEqual(self, self.G.nodes(), [1, 2, 3, 4])
        six.assertCountEqual(self, self.G.edges(), [(1, 2), (1, 3), (1, 4)])
        self.assertEqual(self.G.parent_node, 1)
        six.assertCountEqual(self, self.G.children_nodes, [2, 3, 4])

        self.assertRaises(ValueError, self.G.add_edge, 2, 3)
        self.assertRaises(ValueError, self.G.add_edge, 3, 6)
        self.assertRaises(ValueError, self.G.add_edge, 5, 6)
        self.assertRaises(ValueError, self.G.add_edges_from, [(1, 5), (2, 6)])
        self.assertRaises(ValueError, self.G.add_edges_from, [(2, 6)])

    def test_add_edge_selfloop(self):
        self.assertRaises(ValueError, self.G.add_edge, 'a', 'a')
        self.assertRaises(ValueError, self.G.add_edge, 1, 1)

    def test_add_edges_from_self_loop(self):
        self.assertRaises(ValueError, self.G.add_edges_from,
                          [('a', 'a')])

    def test_update_node_parents_bm_constructor(self):
        self.g = NaiveBayes([('a', 'b'), ('a', 'c')])
        self.assertListEqual(self.g.predecessors('a'), [])
        self.assertListEqual(self.g.predecessors('b'), ['a'])
        self.assertListEqual(self.g.predecessors('c'), ['a'])

    def test_update_node_parents(self):
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edges_from([('a', 'b'), ('a', 'c')])
        self.assertListEqual(self.G.predecessors('a'), [])
        self.assertListEqual(self.G.predecessors('b'), ['a'])
        self.assertListEqual(self.G.predecessors('c'), ['a'])

    def tearDown(self):
        del self.G