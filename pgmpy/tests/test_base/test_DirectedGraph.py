#!/usr/bin/env python3

import unittest

from pgmpy.base import DirectedGraph
import pgmpy.tests.help_functions as hf


class TestDirectedGraphCreation(unittest.TestCase):
    def setUp(self):
        self.graph = DirectedGraph()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, DirectedGraph)

    def test_class_init_with_data_string(self):
        self.graph = DirectedGraph([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.graph.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.graph.edges()),
                             [['a', 'b'], ['b', 'c']])

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

    def test_add_node_weight(self):
        self.graph.add_node('weighted_a', 0.3)
        self.assertEqual(self.graph.node['weighted_a']['weight'], 0.3)

    def test_add_nodes_from_weight(self):
        self.graph.add_nodes_from(['weighted_b', 'weighted_c'], [0.5, 0.6])
        self.assertEqual(self.graph.node['weighted_b']['weight'], 0.5)
        self.assertEqual(self.graph.node['weighted_c']['weight'], 0.6)

        self.graph.add_nodes_from(['e', 'f'])
        self.assertEqual(self.graph.node['e']['weight'], None)
        self.assertEqual(self.graph.node['f']['weight'], None)

    def test_add_edge_string(self):
        self.graph.add_edge('d', 'e')
        self.assertListEqual(sorted(self.graph.nodes()), ['d', 'e'])
        self.assertListEqual(self.graph.edges(), [('d', 'e')])
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

    def test_add_edge_weight(self):
        self.graph.add_edge('a', 'b', weight=0.3)
        self.assertEqual(self.graph.edge['a']['b']['weight'], 0.3)

    def test_add_edges_from_weight(self):
        self.graph.add_edges_from([('b', 'c'), ('c', 'd')], weights=[0.5, 0.6])
        self.assertEqual(self.graph.edge['b']['c']['weight'], 0.5)
        self.assertEqual(self.graph.edge['c']['d']['weight'], 0.6)

        self.graph.add_edges_from([('e', 'f')])
        self.assertEqual(self.graph.edge['e']['f']['weight'], None)

    def test_update_node_parents_bm_constructor(self):
        self.graph = DirectedGraph([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.graph.predecessors('a'), [])
        self.assertListEqual(self.graph.predecessors('b'), ['a'])
        self.assertListEqual(self.graph.predecessors('c'), ['b'])

    def test_update_node_parents(self):
        self.graph.add_nodes_from(['a', 'b', 'c'])
        self.graph.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.graph.predecessors('a'), [])
        self.assertListEqual(self.graph.predecessors('b'), ['a'])
        self.assertListEqual(self.graph.predecessors('c'), ['b'])

    def test_get_leaves(self):
        self.graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'),
                                   ('D', 'E'), ('D', 'F'), ('A', 'G')])
        self.assertEqual(sorted(self.graph.get_leaves()),
                         sorted(['C', 'G', 'E', 'F']))

    def test_get_roots(self):
        self.graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'),
                                   ('D', 'E'), ('D', 'F'), ('A', 'G')])
        self.assertEqual(['A'], self.graph.get_roots())
        self.graph.add_edge('H', 'G')
        self.assertEqual(sorted(['A', 'H']), sorted(self.graph.get_roots()))

    def tearDown(self):
        del self.graph


class TestDirectedGraphMoralization(unittest.TestCase):
    def setUp(self):
        self.graph = DirectedGraph()
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])

    def test_get_parents(self):
        self.assertListEqual(sorted(self.graph.get_parents('grade')),
                             ['diff', 'intel'])

    def test_moralize(self):
        moral_graph = self.graph.moralize()
        self.assertListEqual(hf.recursive_sorted(moral_graph.edges()),
                             [['diff', 'grade'], ['diff', 'intel'],
                              ['grade', 'intel']])

    def tearDown(self):
        del self.graph
