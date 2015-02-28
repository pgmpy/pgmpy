#!/usr/bin/env python3

from pgmpy.base import DirectedGraph
import pgmpy.tests.help_functions as hf
import unittest


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

    def tearDown(self):
        del self.graph


class TestDirectedGraphMoralization(unittest.TestCase):
    def setUp(self):
        self.graph = DirectedGraph()

    def test_get_parents(self):
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.assertListEqual(sorted(self.graph.get_parents('grade')),
                             ['diff', 'intel'])

    def test_moralize(self):
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        moral_graph = self.graph.moralize()
        self.assertListEqual(hf.recursive_sorted(moral_graph.edges()),
                             [['diff', 'grade'], ['diff', 'intel'],
                              ['grade', 'intel']])

    def tearDown(self):
        del self.graph
