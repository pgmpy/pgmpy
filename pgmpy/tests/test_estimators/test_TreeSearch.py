import unittest

import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.estimators import TreeSearch

class TestTreeSearch(unittest.TestCase):
    def setUp(self):
        self.data1 = pd.DataFrame(
            data = [[1, 1, 0], [1, 1, 1], [1, 1, 0]],
            columns=['B', 'C', 'D']
        )

        self.data2 = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, 5)), columns=['A', 'B', 'C', 'D', 'E'])

    def test_estimate(self):
        # learn tree structure using D as root node
        est = TreeSearch(self.data1, root_node='D')
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["B", "C", "D"])
        self.assertCountEqual(
            dag.edges(), [("D", "B"), ("D", "C")]
        )

        # check tree structure exists
        self.assertTrue(dag.has_edge("D", "B"))
        self.assertTrue(dag.has_edge("D", "C"))

        # learn tree structure using B as root node
        est = TreeSearch(self.data1, root_node='B')
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["B", "C", "D"])
        self.assertCountEqual(
            dag.edges(), [("B", "D"), ("D", "C")]
        )

        # check tree structure exists
        self.assertTrue(dag.has_edge("B", "D"))
        self.assertTrue(dag.has_edge("D", "C"))

        # check invalid root node
        est = TreeSearch(self.data1, root_node='A')
        with self.assertRaises(ValueError):
            est.estimate()

        # learn graph structure
        est = TreeSearch(self.data2, root_node='A')
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D", "E"])
        self.assertTrue(nx.is_tree(dag))

    def tearDown(self):
        del self.data1
        del self.data2

