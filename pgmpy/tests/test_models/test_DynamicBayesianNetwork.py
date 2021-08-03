import unittest

import numpy as np
import pandas as pd
import numpy.testing as np_test

import pgmpy.tests.help_functions as hf
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference


class TestDynamicBayesianNetworkCreation(unittest.TestCase):
    def setUp(self):
        self.network = DynamicBayesianNetwork()

    def test_add_single_node(self):
        self.network.add_node("a")
        self.assertListEqual(self.network._nodes(), ["a"])

    def test_add_multiple_nodes(self):
        self.network.add_nodes_from(["a", "b", "c"])
        self.assertListEqual(sorted(self.network._nodes()), ["a", "b", "c"])

    def test_add_single_edge_with_timeslice(self):
        self.network.add_edge(("a", 0), ("b", 0))
        self.assertListEqual(
            sorted(self.network.edges()), [(("a", 0), ("b", 0)), (("a", 1), ("b", 1))]
        )
        self.assertListEqual(sorted(self.network._nodes()), ["a", "b"])

    def test_add_edge_with_different_number_timeslice(self):
        self.network.add_edge(("a", 2), ("b", 2))
        self.assertListEqual(
            sorted(self.network.edges()), [(("a", 0), ("b", 0)), (("a", 1), ("b", 1))]
        )

    def test_add_edge_going_backward(self):
        self.assertRaises(
            NotImplementedError, self.network.add_edge, ("a", 1), ("b", 0)
        )

    def test_add_edge_with_farther_timeslice(self):
        self.assertRaises(ValueError, self.network.add_edge, ("a", 2), ("b", 4))

    def test_add_edge_with_self_loop(self):
        self.assertRaises(ValueError, self.network.add_edge, ("a", 0), ("a", 0))

    def test_add_edge_with_varying_length(self):
        self.assertRaises(ValueError, self.network.add_edge, ("a", 1, 1), ("b", 2))
        self.assertRaises(ValueError, self.network.add_edge, ("b", 2), ("a", 2, 3))

    def test_add_edge_with_closed_path(self):
        self.assertRaises(
            ValueError,
            self.network.add_edges_from,
            [(("a", 0), ("b", 0)), (("b", 0), ("c", 0)), (("c", 0), ("a", 0))],
        )

    def test_add_single_edge_without_timeslice(self):
        self.assertRaises(ValueError, self.network.add_edge, "a", "b")

    def test_add_single_edge_with_incorrect_timeslice(self):
        self.assertRaises(ValueError, self.network.add_edge, ("a", "b"), ("b", "c"))

    def test_add_multiple_edges(self):
        self.network.add_edges_from(
            [(("a", 0), ("b", 0)), (("a", 0), ("a", 1)), (("b", 0), ("b", 1))]
        )
        self.assertListEqual(
            sorted(self.network.edges()),
            [
                (("a", 0), ("a", 1)),
                (("a", 0), ("b", 0)),
                (("a", 1), ("b", 1)),
                (("b", 0), ("b", 1)),
            ],
        )

    def tearDown(self):
        del self.network


class TestDynamicBayesianNetworkMethods(unittest.TestCase):
    def setUp(self):
        self.network = DynamicBayesianNetwork()
        self.grade_cpd = TabularCPD(
            ("G", 0),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 0), ("I", 0)],
            evidence_card=[2, 2],
        )
        self.d_i_cpd = TabularCPD(
            ("D", 1),
            2,
            values=[[0.6, 0.3], [0.4, 0.7]],
            evidence=[("D", 0)],
            evidence_card=[2],
        )
        self.diff_cpd = TabularCPD(("D", 0), 2, values=[[0.6], [0.4]])
        self.intel_cpd = TabularCPD(("I", 0), 2, values=[[0.7], [0.3]])
        self.i_i_cpd = TabularCPD(
            ("I", 1),
            2,
            values=[[0.5, 0.4], [0.5, 0.6]],
            evidence=[("I", 0)],
            evidence_card=[2],
        )
        self.grade_1_cpd = TabularCPD(
            ("G", 1),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 1), ("I", 1)],
            evidence_card=[2, 2],
        )

    def test_get_intra_and_inter_edges(self):
        self.network.add_edges_from(
            [(("a", 0), ("b", 0)), (("a", 0), ("a", 1)), (("b", 0), ("b", 1))]
        )
        self.assertListEqual(
            sorted(self.network.get_intra_edges()), [(("a", 0), ("b", 0))]
        )
        self.assertListEqual(
            sorted(self.network.get_intra_edges(1)), [(("a", 1), ("b", 1))]
        )
        self.assertRaises(ValueError, self.network.get_intra_edges, -1)
        self.assertRaises(ValueError, self.network.get_intra_edges, "-")
        self.assertListEqual(
            sorted(self.network.get_inter_edges()),
            [(("a", 0), ("a", 1)), (("b", 0), ("b", 1))],
        )

    def test_get_interface_nodes(self):
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        self.assertListEqual(
            sorted(self.network.get_interface_nodes()), [("D", 0), ("I", 0)]
        )

        self.assertRaises(ValueError, self.network.get_interface_nodes, -1)
        self.assertRaises(ValueError, self.network.get_interface_nodes, "-")

    def test_get_slice_nodes(self):
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        self.assertListEqual(
            sorted(self.network.get_slice_nodes()), [("D", 0), ("G", 0), ("I", 0)]
        )
        self.assertListEqual(
            sorted(self.network.get_slice_nodes(1)), [("D", 1), ("G", 1), ("I", 1)]
        )
        self.assertRaises(ValueError, self.network.get_slice_nodes, -1)
        self.assertRaises(ValueError, self.network.get_slice_nodes, "-")

    def test_add_single_cpds(self):
        self.network.add_edges_from([(("D", 0), ("G", 0)), (("I", 0), ("G", 0))])
        self.network.add_cpds(self.grade_cpd)
        self.assertListEqual(self.network.get_cpds(), [self.grade_cpd])

    def test_get_cpds(self):
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        self.network.add_cpds(
            self.grade_cpd, self.d_i_cpd, self.diff_cpd, self.intel_cpd, self.i_i_cpd
        )
        self.network.initialize_initial_state()
        self.assertEqual(
            set(self.network.get_cpds()),
            set([self.diff_cpd, self.intel_cpd, self.grade_cpd]),
        )
        self.assertEqual(
            {cpd.variable for cpd in self.network.get_cpds(time_slice=1)},
            {("D", 1), ("I", 1), ("G", 1)},
        )

    def test_add_multiple_cpds(self):
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        self.network.add_cpds(
            self.grade_cpd, self.d_i_cpd, self.diff_cpd, self.intel_cpd, self.i_i_cpd
        )
        self.assertEqual(self.network.get_cpds(("G", 0)).variable, ("G", 0))
        self.assertEqual(self.network.get_cpds(("D", 1)).variable, ("D", 1))
        self.assertEqual(self.network.get_cpds(("D", 0)).variable, ("D", 0))
        self.assertEqual(self.network.get_cpds(("I", 0)).variable, ("I", 0))
        self.assertEqual(self.network.get_cpds(("I", 1)).variable, ("I", 1))

    def test_initialize_initial_state(self):

        self.network.add_nodes_from(["D", "G", "I", "S", "L"])
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        self.network.add_cpds(
            self.grade_cpd, self.d_i_cpd, self.diff_cpd, self.intel_cpd, self.i_i_cpd
        )
        self.network.initialize_initial_state()
        self.assertEqual(len(self.network.cpds), 6)
        self.assertEqual(self.network.get_cpds(("G", 1)).variable, ("G", 1))

    def test_moralize(self):
        self.network.add_edges_from(([(("D", 0), ("G", 0)), (("I", 0), ("G", 0))]))
        moral_graph = self.network.moralize()
        self.assertListEqual(
            hf.recursive_sorted(moral_graph.edges()),
            [
                [("D", 0), ("G", 0)],
                [("D", 0), ("I", 0)],
                [("D", 1), ("G", 1)],
                [("D", 1), ("I", 1)],
                [("G", 0), ("I", 0)],
                [("G", 1), ("I", 1)],
            ],
        )

    def test_copy(self):
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        cpd = TabularCPD(
            ("G", 0),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 0), ("I", 0)],
            evidence_card=[2, 2],
        )
        self.network.add_cpds(cpd)
        copy = self.network.copy()
        self.assertIsInstance(copy, DynamicBayesianNetwork)
        self.assertListEqual(sorted(self.network._nodes()), sorted(copy._nodes()))
        self.assertListEqual(sorted(self.network.edges()), sorted(copy.edges()))
        self.assertListEqual(self.network.get_cpds(), copy.get_cpds())
        self.assertListEqual(
            sorted(self.network.get_intra_edges()), sorted(copy.get_intra_edges())
        )
        self.assertListEqual(
            sorted(self.network.get_inter_edges()), sorted(copy.get_inter_edges())
        )
        self.assertListEqual(
            sorted(self.network.get_slice_nodes()), sorted(copy.get_slice_nodes())
        )

        copy.cpds[0].values = np.array(
            [[0.4, 0.05, 0.3, 0.5], [0.3, 0.25, 0.5, 0.3], [0.3, 0.7, 0.2, 0.2]]
        )
        self.assertNotEqual(self.network.get_cpds(), copy.get_cpds())
        self.network.add_cpds(self.i_i_cpd, self.d_i_cpd)

        copy.add_cpds(self.diff_cpd, self.intel_cpd)
        self.network.add_node("A")
        copy.add_node("Z")
        self.network.add_edge(("A", 0), ("D", 0))
        copy.add_edge(("Z", 0), ("D", 0))
        self.assertNotEqual(sorted(self.network._nodes()), sorted(copy._nodes()))
        self.assertNotEqual(sorted(self.network.edges()), sorted(copy.edges()))
        self.assertNotEqual(self.network.get_cpds(), copy.get_cpds())
        self.assertNotEqual(
            sorted(self.network.get_intra_edges()), sorted(copy.get_intra_edges())
        )
        self.assertListEqual(
            sorted(self.network.get_inter_edges()), sorted(copy.get_inter_edges())
        )
        self.assertNotEqual(
            sorted(self.network.get_slice_nodes()), sorted(copy.get_slice_nodes())
        )

        self.network.add_edge(("A", 0), ("D", 1))
        copy.add_edge(("Z", 0), ("D", 1))
        self.assertNotEqual(
            sorted(self.network.get_inter_edges()), sorted(copy.get_inter_edges())
        )

    def test_fit(self):
        model = DynamicBayesianNetwork(
            [
                (("A", 0), ("B", 0)),
                (("A", 0), ("C", 0)),
                (("B", 0), ("D", 0)),
                (("C", 0), ("D", 0)),
                (("A", 0), ("A", 1)),
                (("B", 0), ("B", 1)),
                (("C", 0), ("C", 1)),
                (("D", 0), ("D", 1)),
            ]
        )

        data = np.random.randint(low=0, high=2, size=(10000, 20))
        colnames = []
        for t in range(5):
            colnames.extend([("A", t), ("B", t), ("C", t), ("D", t)])
        df = pd.DataFrame(data, columns=colnames)
        model.fit(df)

        self.assertTrue(model.check_model())
        self.assertEqual(len(model.cpds), 8)
        for cpd in model.cpds:
            np_test.assert_almost_equal(cpd.values, 0.5, decimal=1)

        self.assertRaises(ValueError, model.fit, df, "bayesian")
        self.assertRaises(ValueError, model.fit, df.values)
        wrong_colnames = []
        for t in range(5):
            wrong_colnames.extend(
                [("A", t + 1), ("B", t + 1), ("C", t + 1), ("D", t + 1)]
            )
        df.columns = wrong_colnames
        self.assertRaises(ValueError, model.fit, df)

    def test_get_markov_blanket(self):
        self.network.add_edges_from(
            [(("a", 0), ("a", 1)), (("a", 0), ("b", 1)), (("b", 0), ("b", 1))]
        )

        markov_blanket = self.network.get_markov_blanket(("a", 1))
        self.assertListEqual(
            markov_blanket,
            [
                ("a", 0),  # parent
                ("a", 1),  # child 1's parent
                ("a", 2),  # child 1
                ("b", 1),  # child 2's parent
                ("b", 2),  # child 2
            ],
        )

    def test_active_trail_nodes(self):
        self.network.add_edges_from(
            [(("a", 0), ("a", 1)), (("a", 0), ("b", 1)), (("b", 0), ("b", 1))]
        )

        active_trail = self.network.active_trail_nodes(("a", 0))
        self.assertListEqual(
            sorted(active_trail.get(("a", 0))), [("a", 0), ("a", 1), ("b", 1)]
        )

        active_trail = self.network.active_trail_nodes(("a", 0), observed=[("b", 1)])
        self.assertListEqual(
            sorted(active_trail.get(("a", 0))), [("a", 0), ("a", 1), ("b", 0)]
        )

    def tearDown(self):
        del self.network


class TestDynamicBayesianNetworkMethods2(unittest.TestCase):
    def setUp(self):
        self.G = DynamicBayesianNetwork()
        self.G.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        """
        G.edges()
        [(('I', 0), ('G', 0)), (('I', 0), ('I', 1)),
         (('D', 1), ('G', 1)), (('D', 0), ('G', 0)),
         (('D', 0), ('D', 1)), (('I', 1), ('G', 1))]

        """

    def test_check_model(self):

        grade_cpd = TabularCPD(
            ("G", 0),
            3,
            values=[[0.3, 0.05, 0.7, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.2, 0.2]],
            evidence=[("D", 0), ("I", 0)],
            evidence_card=[2, 2],
        )

        d_i_cpd = TabularCPD(
            ("D", 1),
            2,
            values=[[0.6, 0.3], [0.4, 0.7]],
            evidence=[("D", 0)],
            evidence_card=[2],
        )

        diff_cpd = TabularCPD(("D", 0), 2, values=[[0.6], [0.4]])

        intel_cpd = TabularCPD(("I", 0), 2, values=[[0.7], [0.3]])

        i_i_cpd = TabularCPD(
            ("I", 1),
            2,
            values=[[0.5, 0.4], [0.5, 0.6]],
            evidence=[("I", 0)],
            evidence_card=[2],
        )

        grade_1_cpd = TabularCPD(
            ("G", 1),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 1), ("I", 1)],
            evidence_card=[2, 2],
        )

        self.G.add_cpds(grade_cpd, d_i_cpd, i_i_cpd)
        self.assertTrue(self.G.check_model())

        self.G.remove_cpds(grade_cpd, d_i_cpd, i_i_cpd)
        self.G.add_cpds(grade_1_cpd, diff_cpd, intel_cpd)
        self.assertTrue(self.G.check_model())

    def test_check_model1(self):

        diff_cpd = TabularCPD(
            ("D", 0),
            3,
            values=[[0.3, 0.05, 0.7, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.2, 0.2]],
            evidence=[("G", 0), ("I", 0)],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(diff_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(diff_cpd)

        grade_cpd = TabularCPD(
            ("G", 0),
            2,
            values=[[0.6, 0.3], [0.4, 0.7]],
            evidence=[("D", 0)],
            evidence_card=[2],
        )
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        diff_cpd = TabularCPD(
            ("D", 0),
            2,
            values=[[0.6, 0.3], [0.4, 0.7]],
            evidence=[("D", 1)],
            evidence_card=[2],
        )
        self.G.add_cpds(diff_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(diff_cpd)

        grade_cpd = TabularCPD(
            ("G", 0),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 1), ("I", 1)],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        grade_cpd = TabularCPD(
            ("G", 1),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 0), ("I", 0)],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        grade_cpd = TabularCPD(
            ("G", 0),
            2,
            values=[[0.6, 0.3], [0.4, 0.7]],
            evidence=[("D", 1)],
            evidence_card=[2],
        )
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

    def test_check_model2(self):

        grade_cpd = TabularCPD(
            ("G", 0),
            3,
            values=[[0.9, 0.05, 0.7, 0.5], [0.4, 0.25, 0.1, 0.3], [0.3, 0.7, 0.2, 0.2]],
            evidence=[("D", 0), ("I", 0)],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        d_i_cpd = TabularCPD(
            ("D", 1),
            2,
            values=[[0.1, 0.3], [0.4, 0.7]],
            evidence=[("D", 0)],
            evidence_card=[2],
        )
        self.G.add_cpds(d_i_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(d_i_cpd)

        diff_cpd = TabularCPD(("D", 0), 2, values=[[0.7], [0.4]])
        self.G.add_cpds(diff_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(diff_cpd)

        intel_cpd = TabularCPD(("I", 0), 2, values=[[1.7], [0.3]])
        self.G.add_cpds(intel_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(intel_cpd)

        i_i_cpd = TabularCPD(
            ("I", 1),
            2,
            values=[[0.9, 0.4], [0.5, 0.6]],
            evidence=[("I", 0)],
            evidence_card=[2],
        )
        self.G.add_cpds(i_i_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(i_i_cpd)

        grade_1_cpd = TabularCPD(
            ("G", 1),
            3,
            values=[[0.3, 0.05, 0.8, 0.5], [0.4, 0.5, 0.1, 0.3], [0.3, 0.7, 0.1, 0.2]],
            evidence=[("D", 1), ("I", 1)],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(grade_1_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_1_cpd)

    def tearDown(self):
        del self.G


class TestDynamicBayesianNetworkMethods3(unittest.TestCase):
    def setUp(self):
        self.cancer_model = DynamicBayesianNetwork()
        #########################    1    ######################
        self.cpd_poll = TabularCPD(
            variable=("Pollution", 0), variable_card=2, values=[[0.9], [0.1]]
        )
        self.cpd_smoke = TabularCPD(
            variable=("Smoker", 0), variable_card=2, values=[[0.3], [0.7]]
        )
        self.cpd_cancer = TabularCPD(
            variable=("Cancer", 0),
            variable_card=2,
            values=[[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]],
            evidence=[("Smoker", 0), ("Pollution", 0)],
            evidence_card=[2, 2],
        )
        self.cpd_xray = TabularCPD(
            variable=("Xray", 0),
            variable_card=2,
            values=[[0.9, 0.2], [0.1, 0.8]],
            evidence=[("Cancer", 0)],
            evidence_card=[2],
        )
        self.cpd_dysp = TabularCPD(
            variable=("Dyspnoea", 0),
            variable_card=2,
            values=[[0.65, 0.3], [0.35, 0.7]],
            evidence=[("Cancer", 0)],
            evidence_card=[2],
        )

        #########################    2    ######################
        self.cpd_poll2 = TabularCPD(
            variable=("Pollution", 1),
            variable_card=2,
            values=[[0.9, 0.1], [0.1, 0.9]],
            evidence=[("Pollution", 0)],
            evidence_card=[2],
        )
        self.cpd_smoke2 = TabularCPD(
            variable=("Smoker", 1),
            variable_card=2,
            values=[[0.7, 0.3], [0.3, 0.7]],
            evidence=[("Smoker", 0)],
            evidence_card=[2],
        )
        self.cpd_cancer2 = TabularCPD(
            variable=("Cancer", 1),
            variable_card=2,
            values=[[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]],
            evidence=[("Smoker", 1), ("Pollution", 1)],
            evidence_card=[2, 2],
        )
        self.cpd_xray2 = TabularCPD(
            variable=("Xray", 1),
            variable_card=2,
            values=[[0.9, 0.2], [0.1, 0.8]],
            evidence=[("Cancer", 1)],
            evidence_card=[2],
        )
        self.cpd_dysp2 = TabularCPD(
            variable=("Dyspnoea", 1),
            variable_card=2,
            values=[[0.65, 0.3], [0.35, 0.7]],
            evidence=[("Cancer", 1)],
            evidence_card=[2],
        )

    def test_initialize_and_infer1(self):
        self.cancer_model.add_edges_from(
            [
                (("Pollution", 0), ("Cancer", 0)),
                (("Smoker", 0), ("Cancer", 0)),
                (("Cancer", 0), ("Xray", 0)),
                (("Cancer", 0), ("Dyspnoea", 0)),
            ]
        )

        self.cancer_model.add_cpds(
            self.cpd_poll, self.cpd_smoke, self.cpd_cancer, self.cpd_xray, self.cpd_dysp
        )
        self.cancer_model.initialize_initial_state()

        self.assertEqual(len(self.cancer_model.cpds), 10)

        self.cancer_inf = DBNInference(self.cancer_model)
        self.cancer_query_result = self.cancer_inf.query(
            [("Xray", 0)], {("Smoker", 0): 0}
        )[("Xray", 0)].values

        self.assertAlmostEqual(self.cancer_query_result[0], 0.2224, 4)
        self.assertAlmostEqual(self.cancer_query_result[1], 0.7776, 4)

    def test_initialize_and_infer2(self):
        self.cancer_model.add_edges_from(
            [
                (("Pollution", 0), ("Cancer", 0)),
                (("Smoker", 0), ("Cancer", 0)),
                (("Cancer", 0), ("Xray", 0)),
                (("Cancer", 0), ("Dyspnoea", 0)),
                (("Pollution", 0), ("Pollution", 1)),
                (("Smoker", 0), ("Smoker", 1)),
                (("Pollution", 1), ("Cancer", 1)),
                (("Smoker", 1), ("Cancer", 1)),
                (("Cancer", 1), ("Xray", 1)),
                (("Cancer", 1), ("Dyspnoea", 1)),
            ]
        )

        self.cancer_model.add_cpds(
            self.cpd_poll,
            self.cpd_smoke,
            self.cpd_cancer,
            self.cpd_xray,
            self.cpd_dysp,
            self.cpd_poll2,
            self.cpd_smoke2,
            self.cpd_cancer2,
            self.cpd_xray2,
            self.cpd_dysp2,
        )
        self.cancer_model.initialize_initial_state()

        self.assertEqual(len(self.cancer_model.cpds), 10)

        self.cancer_inf = DBNInference(self.cancer_model)
        self.cancer_query_result = self.cancer_inf.query(
            [("Xray", 1)], {("Smoker", 0): 0}
        )[("Xray", 1)].values

        self.assertAlmostEqual(self.cancer_query_result[0], 0.213307, 4)
        self.assertAlmostEqual(self.cancer_query_result[1], 0.786693, 4)

    def test_initialize_and_infer3(self):
        self.cancer_model.add_edges_from(
            [
                (("Pollution", 0), ("Cancer", 0)),
                (("Smoker", 0), ("Cancer", 0)),
                (("Cancer", 0), ("Xray", 0)),
                (("Cancer", 0), ("Dyspnoea", 0)),
                (("Pollution", 0), ("Pollution", 1)),
                (("Smoker", 0), ("Smoker", 1)),
                (("Pollution", 1), ("Cancer", 1)),
                (("Smoker", 1), ("Cancer", 1)),
                (("Cancer", 1), ("Xray", 1)),
                (("Cancer", 1), ("Dyspnoea", 1)),
            ]
        )

        self.cancer_model.add_cpds(
            self.cpd_poll,
            self.cpd_smoke,
            self.cpd_cancer,
            self.cpd_xray,
            self.cpd_dysp,
            self.cpd_poll2,
            self.cpd_smoke2,
            self.cpd_cancer2,
            self.cpd_xray2,
            self.cpd_dysp2,
        )
        self.cancer_model.initialize_initial_state()

        self.assertEqual(len(self.cancer_model.cpds), 10)

        self.cancer_inf = DBNInference(self.cancer_model)
        self.cancer_query_result = self.cancer_inf.query(
            [("Xray", 2)], {("Smoker", 0): 0}
        )[("Xray", 2)].values

        self.assertAlmostEqual(self.cancer_query_result[0], 0.2158, 4)

        self.cancer_query_result = self.cancer_inf.query(
            [("Dyspnoea", 3)], {("Pollution", 0): 0}
        )[("Dyspnoea", 3)].values
        self.assertAlmostEqual(self.cancer_query_result[0], 0.3070, 4)

    def tearDown(self):
        del self.cancer_model
