import unittest

import pandas as pd

from pgmpy.estimators import TreeAugmentedNaiveBayesSearch


class TestTreeAugmentedNaiveBayesSearch(unittest.TestCase):
    def setUp(self):
        self.data1 = pd.DataFrame(
            data=[[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]],
            columns=["A", "B", "C", "D"],
        )

        self.data2 = pd.DataFrame(data=[[0, 1], [1, 1], [0, 1]], columns=["A", "B"])

    def test_estimate(self):
        # learn graph structure
        est = TreeAugmentedNaiveBayesSearch(self.data1, class_node="A", root_node="D")
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D"])
        self.assertCountEqual(
            dag.edges(), [("A", "B"), ("A", "C"), ("A", "D"), ("D", "B"), ("D", "C")]
        )

        # check directed edge between dependent and independent variables
        self.assertTrue(dag.has_edge("A", "B"))
        self.assertTrue(dag.has_edge("A", "C"))
        self.assertTrue(dag.has_edge("A", "D"))

        # check tree structure exists over independent variables
        self.assertTrue(dag.has_edge("D", "B"))
        self.assertTrue(dag.has_edge("D", "C"))

        # check invalid class node
        est = TreeAugmentedNaiveBayesSearch(self.data1, class_node="X", root_node="D")
        with self.assertRaises(ValueError):
            est.estimate()

        # check invalid root node
        est = TreeAugmentedNaiveBayesSearch(self.data1, class_node="A", root_node="X")
        with self.assertRaises(ValueError):
            est.estimate()

        # learn graph structure
        est = TreeAugmentedNaiveBayesSearch(self.data2, class_node="A")
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["A", "B"])
        self.assertCountEqual(dag.edges(), [("A", "B")])

    def tearDown(self):
        del self.data1
        del self.data2
