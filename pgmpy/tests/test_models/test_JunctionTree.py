from pgmpy.models import JunctionTree
from pgmpy.tests import help_functions as hf
import unittest


class TestJunctionTreeCreation(unittest.TestCase):
    def setUp(self):
        self.graph = JunctionTree()

    def test_add_single_node(self):
        self.graph.add_node(('a', 'b'))
        self.assertListEqual(self.graph.nodes(), [('a', 'b')])

    def test_add_single_node_raises_error(self):
        self.assertRaises(TypeError, self.graph.add_node, 'a')

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(hf.recursive_sorted(self.graph.nodes()),
                             [['a', 'b'], ['b', 'c']])

    def test_add_single_edge(self):
        self.graph.add_edge(('a', 'b'), ('b', 'c'))
        self.assertListEqual(hf.recursive_sorted(self.graph.nodes()),
                             [['a', 'b'], ['b', 'c']])
        self.assertListEqual(sorted([node for edge in self.graph.edges()
                                     for node in edge]),
                             [('a', 'b'), ('b', 'c')])

    def test_add_single_edge_raises_error(self):
        self.assertRaises(ValueError, self.graph.add_edge,
                          ('a', 'b'), ('c', 'd'))

    def test_add_cyclic_path_raises_error(self):
        self.graph.add_edge(('a', 'b'), ('b', 'c'))
        self.graph.add_edge(('b', 'c'), ('c', 'd'))
        self.assertRaises(ValueError, self.graph.add_edge, ('c', 'd'), ('a', 'b'))

    def tearDown(self):
        del self.graph
