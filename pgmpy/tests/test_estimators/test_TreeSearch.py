import unittest

import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling


class TestTreeSearch(unittest.TestCase):
    def setUp(self):
        # test data for chow-liu
        self.data1 = pd.DataFrame(
            data=[[1, 1, 0], [1, 1, 1], [1, 1, 0]], columns=["B", "C", "D"]
        )

        # test data for chow-liu
        self.data2 = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(100, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        # test data for TAN
        self.data3 = pd.DataFrame(
            data=[[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]],
            columns=["A", "B", "C", "D"],
        )

        # test data for TAN
        model = BayesianModel([('A', 'R'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('R', 'B'), ('R', 'C'), ('R', 'D'), ('R', 'E')])
        cpd_a = TabularCPD('A', 2, [[0.7], [0.3]])
        cpd_r = TabularCPD('R',3,[[0.6,0.2],[0.3,0.5],[0.1,0.3]],evidence=['A'], evidence_card=[2])
        cpd_b = TabularCPD('B',3,[[0.1,0.1,0.2,0.2,0.7,0.1],
                                  [0.1,0.3,0.1,0.2,0.1,0.2],
                                  [0.8,0.6,0.7,0.6,0.2,0.7]],
                                  evidence=['A','R'], evidence_card=[2,3])
        cpd_c = TabularCPD('C',2,[[0.7,0.2,0.2,0.5,0.1,0.3],
                                  [0.3,0.8,0.8,0.5,0.9,0.7]],
                                  evidence=['A','R'], evidence_card=[2,3])
        cpd_d = TabularCPD('D',3,[[0.3,0.8,0.2,0.8,0.4,0.7],
                                  [0.4,0.1,0.4,0.1,0.1,0.1],
                                  [0.3,0.1,0.4,0.1,0.5,0.2]],
                                  evidence=['A','R'], evidence_card=[2,3])
        cpd_e = TabularCPD('E',2,[[0.5,0.6,0.6,0.5,0.5,0.4],
                                  [0.5,0.4,0.4,0.5,0.5,0.6]],
                                  evidence=['A','R'], evidence_card=[2,3])
        model.add_cpds(cpd_a, cpd_r, cpd_b, cpd_c, cpd_d, cpd_e)
        inference = BayesianModelSampling(model)
        self.data4 = inference.forward_sample(size=10000, return_type='dataframe')

    def test_estimate_chow_liu(self):
        # learn tree structure using D as root node
        est = TreeSearch(self.data1, root_node="D", return_type='chow-liu')

        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["B", "C", "D"])
        self.assertCountEqual(dag.edges(), [("D", "B"), ("D", "C")])

        # check tree structure exists
        self.assertTrue(dag.has_edge("D", "B"))
        self.assertTrue(dag.has_edge("D", "C"))

        # learn tree structure using B as root node
        est = TreeSearch(self.data1, root_node="B", return_type='chow-liu')

        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["B", "C", "D"])
        self.assertCountEqual(dag.edges(), [("B", "D"), ("D", "C")])

        # check tree structure exists
        self.assertTrue(dag.has_edge("B", "D"))
        self.assertTrue(dag.has_edge("D", "C"))

        # check invalid root node
        est = TreeSearch(self.data1, root_node="A", return_type='chow-liu')
        with self.assertRaises(ValueError):
            est.estimate()

        # learn graph structure
        est = TreeSearch(self.data2, root_node="A", return_type='chow-liu')
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D", "E"])
        self.assertTrue(nx.is_tree(dag))

    def test_estimate_tan(self):
        # learn graph structure
        est = TreeSearch(self.data3, root_node="D", return_type='tan', class_node="A")
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

        # check invalid root node
        est = TreeSearch(self.data3, root_node="X", return_type='tan', class_node="A")
        with self.assertRaises(ValueError):
            est.estimate()

        # check invalid class node
        est = TreeSearch(self.data3, root_node="D", return_type='tan', class_node="X")
        with self.assertRaises(ValueError):
            est.estimate()

        est = TreeSearch(self.data3, root_node="D", return_type='tan', class_node="D")
        with self.assertRaises(ValueError):
            est.estimate()

        # learn graph structure
        est = TreeSearch(self.data4, root_node='R', return_type='tan', class_node='A')
        dag = est.estimate()

        # check number of nodes and edges are as expected
        self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D", "E", "R"])
        self.assertCountEqual(
            dag.edges(),
            [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "R"), ("R", "B"), ("R", "C"), ("R", "D"), ("R", "E")]
        )

        # check directed edge between class and independent variables
        self.assertTrue(dag.has_edge("A", "B"))
        self.assertTrue(dag.has_edge("A", "C"))
        self.assertTrue(dag.has_edge("A", "D"))
        self.assertTrue(dag.has_edge("A", "E"))

        # check tree structure exists over independent variables
        self.assertTrue(dag.has_edge("R", "B"))
        self.assertTrue(dag.has_edge("R", "C"))
        self.assertTrue(dag.has_edge("R", "D"))
        self.assertTrue(dag.has_edge("R", "E"))

    def tearDown(self):
        del self.data1
        del self.data2
        del self.data3
        del self.data4
