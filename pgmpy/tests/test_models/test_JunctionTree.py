import unittest
import numpy as np

from pgmpy.factors import Factor
from pgmpy.models import JunctionTree
from pgmpy.tests import help_functions as hf


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


class TestJunctionTreeMethods(unittest.TestCase):
    def setUp(self):
        self.factor1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        self.factor2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        self.factor3 = Factor(['d', 'e'], [2, 2], np.random.rand(4))
        self.factor4 = Factor(['e', 'f'], [2, 2], np.random.rand(4))
        self.factor5 = Factor(['a', 'b', 'e'], [2, 2, 2], np.random.rand(8))

        self.graph1 = JunctionTree()
        self.graph1.add_edge(('a', 'b'), ('b', 'c'))
        self.graph1.add_factors(self.factor1, self.factor2)

        self.graph2 = JunctionTree()
        self.graph2.add_nodes_from([('a', 'b'), ('b', 'c'), ('d', 'e')])
        self.graph2.add_edge(('a', 'b'), ('b', 'c'))
        self.graph2.add_factors(self.factor1, self.factor2, self.factor3)

        self.graph3 = JunctionTree()
        self.graph3.add_edges_from([(('a', 'b'), ('b', 'c')), (('d', 'e'), ('e', 'f'))])
        self.graph3.add_factors(self.factor1, self.factor2, self.factor3, self.factor4)

        self.graph4 = JunctionTree()
        self.graph4.add_edges_from([(('a', 'b', 'e'), ('b', 'c')), (('a', 'b', 'e'), ('e', 'f')),
                                    (('d', 'e'), ('e', 'f'))])
        self.graph4.add_factors(self.factor5, self.factor2, self.factor3, self.factor4)

    def test_check_model(self):
        self.assertRaises(ValueError, self.graph2.check_model)
        self.assertRaises(ValueError, self.graph3.check_model)
        self.assertTrue(self.graph1.check_model())
        self.assertTrue(self.graph4.check_model())

    def tearDown(self):
        del self.factor1
        del self.factor2
        del self.factor3
        del self.factor4
        del self.factor5

        del self.graph1
        del self.graph2
        del self.graph3
        del self.graph4
