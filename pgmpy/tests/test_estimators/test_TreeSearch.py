import unittest

import networkx as nx
import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TestTreeSearch(unittest.TestCase):
    def setUp(self):
        # set random seed
        np.random.seed(0)

        # test data for chow-liu
        self.data12 = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(100, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        # test data for chow-liu
        model = BayesianNetwork(
            [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]
        )
        cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
        cpd_b = TabularCPD(
            "B",
            3,
            [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
            evidence=["A"],
            evidence_card=[2],
        )
        cpd_c = TabularCPD(
            "C", 2, [[0.3, 0.4], [0.7, 0.6]], evidence=["A"], evidence_card=[2]
        )
        cpd_d = TabularCPD(
            "D",
            3,
            [[0.5, 0.3, 0.1], [0.4, 0.4, 0.8], [0.1, 0.3, 0.1]],
            evidence=["B"],
            evidence_card=[3],
        )
        cpd_e = TabularCPD(
            "E",
            2,
            [[0.3, 0.5, 0.2], [0.7, 0.5, 0.8]],
            evidence=["B"],
            evidence_card=[3],
        )
        cpd_f = TabularCPD(
            "F",
            3,
            [[0.3, 0.6], [0.5, 0.2], [0.2, 0.2]],
            evidence=["C"],
            evidence_card=[2],
        )

        model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e, cpd_f)
        inference = BayesianModelSampling(model)
        self.data13 = inference.forward_sample(size=10000)

        # test data for TAN
        model = BayesianNetwork(
            [
                ("A", "R"),
                ("A", "B"),
                ("A", "C"),
                ("A", "D"),
                ("A", "E"),
                ("R", "B"),
                ("R", "C"),
                ("R", "D"),
                ("R", "E"),
            ]
        )
        cpd_a = TabularCPD("A", 2, [[0.7], [0.3]])
        cpd_r = TabularCPD(
            "R",
            3,
            [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
            evidence=["A"],
            evidence_card=[2],
        )
        cpd_b = TabularCPD(
            "B",
            3,
            [
                [0.1, 0.1, 0.2, 0.2, 0.7, 0.1],
                [0.1, 0.3, 0.1, 0.2, 0.1, 0.2],
                [0.8, 0.6, 0.7, 0.6, 0.2, 0.7],
            ],
            evidence=["A", "R"],
            evidence_card=[2, 3],
        )
        cpd_c = TabularCPD(
            "C",
            2,
            [[0.7, 0.2, 0.2, 0.5, 0.1, 0.3], [0.3, 0.8, 0.8, 0.5, 0.9, 0.7]],
            evidence=["A", "R"],
            evidence_card=[2, 3],
        )
        cpd_d = TabularCPD(
            "D",
            3,
            [
                [0.3, 0.8, 0.2, 0.8, 0.4, 0.7],
                [0.4, 0.1, 0.4, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.4, 0.1, 0.5, 0.2],
            ],
            evidence=["A", "R"],
            evidence_card=[2, 3],
        )
        cpd_e = TabularCPD(
            "E",
            2,
            [[0.5, 0.6, 0.6, 0.5, 0.5, 0.4], [0.5, 0.4, 0.4, 0.5, 0.5, 0.6]],
            evidence=["A", "R"],
            evidence_card=[2, 3],
        )
        model.add_cpds(cpd_a, cpd_r, cpd_b, cpd_c, cpd_d, cpd_e)
        inference = BayesianModelSampling(model)
        self.data22 = inference.forward_sample(size=10000)

    def test_estimate_chow_liu(self):
        # learn tree structure using D as root node
        for weight_fn in [
            "mutual_info",
            "adjusted_mutual_info",
            "normalized_mutual_info",
        ]:
            for n_jobs in [2, 1]:
                # learn graph structure
                est = TreeSearch(self.data12, root_node="A", n_jobs=n_jobs)
                dag = est.estimate(
                    estimator_type="chow-liu",
                    edge_weights_fn=weight_fn,
                    show_progress=False,
                )

                # check number of nodes and edges are as expected
                self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D", "E"])
                self.assertTrue(nx.is_tree(dag))

                # learn tree structure using A as root node
                est = TreeSearch(self.data13, root_node="A", n_jobs=n_jobs)
                dag = est.estimate(
                    estimator_type="chow-liu",
                    edge_weights_fn=weight_fn,
                    show_progress=False,
                )

                # check number of nodes and edges are as expected
                self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D", "E", "F"])
                self.assertCountEqual(
                    dag.edges(),
                    [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")],
                )

                # check tree structure exists
                self.assertTrue(dag.has_edge("A", "B"))
                self.assertTrue(dag.has_edge("A", "C"))
                self.assertTrue(dag.has_edge("B", "D"))
                self.assertTrue(dag.has_edge("B", "E"))
                self.assertTrue(dag.has_edge("C", "F"))

    def test_estimate_tan(self):
        for weight_fn in [
            "mutual_info",
            "adjusted_mutual_info",
            "normalized_mutual_info",
        ]:
            for n_jobs in [2, 1]:
                # learn graph structure
                est = TreeSearch(self.data22, root_node="R", n_jobs=n_jobs)
                dag = est.estimate(
                    estimator_type="tan",
                    class_node="A",
                    edge_weights_fn=weight_fn,
                    show_progress=False,
                )

                # check number of nodes and edges are as expected
                self.assertCountEqual(dag.nodes(), ["A", "B", "C", "D", "E", "R"])
                self.assertCountEqual(
                    dag.edges(),
                    [
                        ("A", "B"),
                        ("A", "C"),
                        ("A", "D"),
                        ("A", "E"),
                        ("A", "R"),
                        ("R", "B"),
                        ("R", "C"),
                        ("R", "D"),
                        ("R", "E"),
                    ],
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

    def test_estimate_chow_liu_auto_root_node(self):
        # learn tree structure using auto root node
        est = TreeSearch(self.data12)

        # root node selection
        weights = est._get_weights(self.data12)
        sum_weights = weights.sum(axis=0)
        maxw_idx = np.argsort(sum_weights)[::-1]
        root_node = self.data12.columns[maxw_idx[0]]

        dag = est.estimate(estimator_type="chow-liu", show_progress=False)
        nodes = list(dag.nodes())
        np.testing.assert_equal(nodes[0], root_node)
        np.testing.assert_array_equal(nodes, ["D", "A", "C", "B", "E"])

    def test_estimate_tan_auto_class_node(self):
        # learn tree structure using auto root and class node
        est = TreeSearch(self.data22)

        # root and class node selection
        weights = est._get_weights(self.data22)
        sum_weights = weights.sum(axis=0)
        maxw_idx = np.argsort(sum_weights)[::-1]
        root_node = self.data22.columns[maxw_idx[0]]
        class_node = self.data22.columns[maxw_idx[1]]

        dag = est.estimate(
            estimator_type="tan", class_node=class_node, show_progress=False
        )
        nodes = list(dag.nodes())
        self.assertEqual(nodes[0], root_node)
        self.assertEqual(nodes[-1], class_node)
        self.assertEqual(sorted(nodes), sorted(["C", "R", "A", "D", "E", "B"]))

    def tearDown(self):
        del self.data12
        del self.data22

        get_reusable_executor().shutdown(wait=True)


class TestTreeSearchRealDataSet(unittest.TestCase):
    def setUp(self):
        self.alarm_df = get_example_model("alarm").simulate(int(1e4), seed=42)

    def test_tan(self):
        # Expected values taken from bnlearn.
        expected_edges = [
            ("CVP", "LVFAILURE"),
            ("CVP", "INTUBATION"),
            ("CVP", "TPR"),
            ("CVP", "DISCONNECT"),
            ("CVP", "VENTMACH"),
            ("CVP", "HR"),
            ("CVP", "FIO2"),
            ("CVP", "HRBP"),
            ("CVP", "VENTLUNG"),
            ("CVP", "PAP"),
            ("CVP", "HISTORY"),
            ("CVP", "PCWP"),
            ("CVP", "INSUFFANESTH"),
            ("CVP", "SAO2"),
            ("CVP", "EXPCO2"),
            ("CVP", "PRESS"),
            ("CVP", "PULMEMBOLUS"),
            ("CVP", "ARTCO2"),
            ("CVP", "MINVOLSET"),
            ("LVFAILURE", "HISTORY"),
            ("LVFAILURE", "PCWP"),
            ("INTUBATION", "INSUFFANESTH"),
            ("EXPCO2", "INTUBATION"),
            ("HR", "TPR"),
            ("PRESS", "DISCONNECT"),
            ("VENTLUNG", "VENTMACH"),
            ("VENTMACH", "PRESS"),
            ("VENTMACH", "MINVOLSET"),
            ("HR", "HRBP"),
            ("ARTCO2", "HR"),
            ("SAO2", "FIO2"),
            ("VENTLUNG", "PAP"),
            ("PCWP", "VENTLUNG"),
            ("VENTLUNG", "EXPCO2"),
            ("VENTLUNG", "ARTCO2"),
            ("PAP", "PULMEMBOLUS"),
            ("ARTCO2", "SAO2"),
        ]
        features = [
            "LVFAILURE",
            "INTUBATION",
            "TPR",
            "DISCONNECT",
            "VENTMACH",
            "HR",
            "FIO2",
            "HRBP",
            "VENTLUNG",
            "PAP",
            "HISTORY",
            "PCWP",
            "INSUFFANESTH",
            "SAO2",
            "EXPCO2",
            "PRESS",
            "PULMEMBOLUS",
            "ARTCO2",
            "MINVOLSET",
        ]
        target = "CVP"
        est = TreeSearch(self.alarm_df[features + [target]], root_node=features[0])
        edges = est.estimate(
            estimator_type="tan", class_node=target, show_progress=False
        ).edges()
        self.assertEqual(set(expected_edges), set(edges))

    def tearDown(self):
        get_reusable_executor().shutdown(wait=True)
