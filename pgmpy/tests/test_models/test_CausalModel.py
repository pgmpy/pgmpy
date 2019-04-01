import unittest

import networkx as nx

from pgmpy.models import CausalModel
import pgmpy.tests.help_functions as hf


class TestBaseModelCreation(unittest.TestCase):

    def setUp(self):
        self.G = CausalModel()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = CausalModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.g.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.g.edges()),
                             [['a', 'b'], ['b', 'c']])

    def test_class_init_with_data_nonstring(self):
        CausalModel([(1, 2), (2, 3)])

    def test_add_node_string(self):
        self.G.add_node('a')
        self.assertListEqual(list(self.G.nodes()), ['a'])

    def test_add_node_nonstring(self):
        self.G.add_node(1)

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(['a', 'b', 'c', 'd'])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c', 'd'])

    def test_add_nodes_from_non_string(self):
        self.G.add_nodes_from([1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge('d', 'e')
        self.assertListEqual(sorted(self.G.nodes()), ['d', 'e'])
        self.assertListEqual(list(self.G.edges()), [('d', 'e')])
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edge('a', 'b')
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['a', 'b'], ['d', 'e']])

    def test_add_edge_nonstring(self):
        self.G.add_edge(1, 2)

    def test_add_edge_selfloop(self):
        self.assertRaises(ValueError, self.G.add_edge, 'a', 'a')

    def test_add_edge_result_cycle(self):
        self.G.add_edges_from([('a', 'b'), ('a', 'c')])
        self.assertRaises(ValueError, self.G.add_edge, 'c', 'a')

    def test_add_edges_from_string(self):
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.G.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             [['a', 'b'], ['b', 'c']])
        self.G.add_nodes_from(['d', 'e', 'f'])
        self.G.add_edges_from([('d', 'e'), ('e', 'f')])
        self.assertListEqual(sorted(self.G.nodes()),
                             ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertListEqual(hf.recursive_sorted(self.G.edges()),
                             hf.recursive_sorted([('a', 'b'), ('b', 'c'),
                                                  ('d', 'e'), ('e', 'f')]))

    def test_add_edges_from_nonstring(self):
        self.G.add_edges_from([(1, 2), (2, 3)])

    def test_add_edges_from_self_loop(self):
        self.assertRaises(ValueError, self.G.add_edges_from,
                          [('a', 'a')])

    def test_add_edges_from_result_cycle(self):
        self.assertRaises(ValueError, self.G.add_edges_from,
                          [('a', 'b'), ('b', 'c'), ('c', 'a')])

    def test_update_node_parents_bm_constructor(self):
        self.g = CausalModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(list(self.g.predecessors('a')), [])
        self.assertListEqual(list(self.g.predecessors('b')), ['a'])
        self.assertListEqual(list(self.g.predecessors('c')), ['b'])

    def test_update_node_parents(self):
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(list(self.G.predecessors('a')), [])
        self.assertListEqual(list(self.G.predecessors('b')), ['a'])
        self.assertListEqual(list(self.G.predecessors('c')), ['b'])

    def tearDown(self):
        del self.G


class TestBackdoorPaths(unittest.TestCase):
    """
    These tests are drawn from games presented in The Book of Why and
    Causality by Judea Pearl.  They are small enough to be easy to 
    confirm by hand.
    """
    def test_game1(self):
        game1 = CausalModel([('X', 'A'),
                             ('A', 'Y'),
                             ('A', 'B')])
        deconfounders = game1.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [])

    def test_game2(self):
        game2 = CausalModel([('X', 'E'),
                            ('E', 'Y'),
                            ('A', 'X'),
                            ('A', 'B'),
                            ('B', 'C'),
                            ('D', 'B'),
                            ('D', 'E')])
        deconfounders = game2.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [])

    def test_game3(self):
        game3 = CausalModel([('X', 'Y'),
                            ('X', 'A'),
                            ('B', 'A'),
                            ('B', 'Y'),
                            ('B', 'X')])
        deconfounders = game3.get_deconfounders(treatment="X",
                                                outcome="Y",
                                                maxdepth=1)
        self.assertEqual(deconfounders, [('B',)])

    def test_game4(self):
        game4 = CausalModel([('A', 'X'),
                              ('A', 'B'),
                              ('C', 'B'),
                              ('C', 'Y')])
        deconfounders = game4.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [])

    def test_game5(self):
        game5 = CausalModel([('A', 'X'),
                            ('A', 'B'),
                            ('C', 'B'),
                            ('C', 'Y'),
                            ('X', 'Y'),
                            ('B', 'X')])
        deconfounders = game5.get_deconfounders(treatment="X",
                                                outcome="Y",
                                                maxdepth=2)
        self.assertEqual(deconfounders, [('C',), ('A', 'B'), ('A', 'C'), ('B', 'C')])

    def test_game6(self):
        game6 = CausalModel([('A', 'X'),
                            ('C', 'B'),
                            ('C', 'Y'),
                            ('X', 'Y'),
                            ('B', 'X'),
                            ("A", "Y")])
        deconfounders = game6.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [('A', 'B'), ('A', 'C'), ('A', 'B', 'C')])

    def test_game7(self):
        game7 = CausalModel([('X', 'A'),
                             ('C', 'B'),
                             ('C', 'Y'),
                             ('X', 'Y'),
                             ('B', 'X'),
                             ("A", "Y")])
        deconfounders = game7.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [('B',), ('C',), ('B', 'C')])

    def test_game8(self):
        game8 = CausalModel([('A', 'C'),
                            ('A', 'D'),
                            ('B', 'D'),
                            ('B', 'E'),
                            ('C', 'X'),
                            ("X", "F"),
                            ("F", "Y"),
                            ("E", "Y"),
                            ("D", "X"),
                            ("D", "Y")])
        deconfounders = game8.get_deconfounders(treatment="X", outcome="Y")
        self.assertEqual(deconfounders, [('E', 'D'),
                                        ('C', 'D'),
                                        ('B', 'D'),
                                        ('A', 'D'),
                                        ('E', 'C', 'D'),
                                        ('E', 'B', 'D'),
                                        ('E', 'A', 'D'),
                                        ('C', 'B', 'D'),
                                        ('C', 'A', 'D'),
                                        ('B', 'A', 'D'),
                                        ('E', 'C', 'B', 'D'),
                                        ('E', 'C', 'A', 'D'),
                                        ('E', 'B', 'A', 'D'),
                                        ('C', 'B', 'A', 'D'),
                                        ('E', 'C', 'B', 'A', 'D')])
