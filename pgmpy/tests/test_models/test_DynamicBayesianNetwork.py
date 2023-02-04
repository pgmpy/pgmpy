import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

import pgmpy.tests.help_functions as hf
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, DBNInference, VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.models import DynamicBayesianNetwork as DBN


class TestDynamicBayesianNetworkCreation(unittest.TestCase):
    def setUp(self):
        self.network = DBN()

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
        self.network = DBN()
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

    def test_get_constant_bn(self):
        self.network.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
                (("D", 1), ("G", 1)),
                (("I", 1), ("G", 1)),
            ]
        )
        self.network.add_cpds(
            self.grade_cpd,
            self.d_i_cpd,
            self.diff_cpd,
            self.intel_cpd,
            self.i_i_cpd,
            self.grade_1_cpd,
        )

        bn = self.network.get_constant_bn(t_slice=0)
        self.assertEqual(set(bn.nodes()), {"D_0", "I_0", "G_0", "D_1", "I_1", "G_1"})
        self.assertEqual(
            set(bn.edges()),
            {
                ("D_0", "G_0"),
                ("I_0", "G_0"),
                ("D_0", "D_1"),
                ("I_0", "I_1"),
                ("D_1", "G_1"),
                ("I_1", "G_1"),
            },
        )
        self.assertTrue(bn.check_model())

        bn = self.network.get_constant_bn(t_slice=1)
        self.assertEqual(set(bn.nodes()), {"D_1", "I_1", "G_1", "D_2", "I_2", "G_2"})
        self.assertEqual(
            set(bn.edges()),
            {
                ("D_1", "G_1"),
                ("I_1", "G_1"),
                ("D_1", "D_2"),
                ("I_1", "I_2"),
                ("D_2", "G_2"),
                ("I_2", "G_2"),
            },
        )
        self.assertTrue(bn.check_model())

        bn = self.network.get_constant_bn(t_slice=2)
        self.assertEqual(set(bn.nodes()), {"D_2", "I_2", "G_2", "D_3", "I_3", "G_3"})
        self.assertEqual(
            set(bn.edges()),
            {
                ("D_2", "G_2"),
                ("I_2", "G_2"),
                ("D_2", "D_3"),
                ("I_2", "I_3"),
                ("D_3", "G_3"),
                ("I_3", "G_3"),
            },
        )
        self.assertTrue(bn.check_model())

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

    def test_get_interface_nodes_1node2edges_divergent_edges(self):
        # divergent interface edges from one node (a0->a1,b1). issue #1364
        self.network.add_edges_from(
            [
                (("A", 0), ("A", 1)),
                (("A", 0), ("B", 0)),
                (("A", 0), ("B", 1)),
                (("A", 1), ("B", 1)),
            ]
        )
        self.assertListEqual(self.network.get_interface_nodes(0), [("A", 0), ("A", 0)])
        self.assertListEqual(self.network.get_interface_nodes(1), [("A", 1), ("B", 1)])

    def test_get_interface_nodes_convergent_edges(self):
        # convergent interface edges to one node(a0,b0->b1). issue #1364
        self.network.add_edges_from(
            [
                (("A", 0), ("B", 1)),
                (("B", 0), ("B", 1)),
                (("A", 0), ("B", 0)),
                (("A", 1), ("B", 1)),
            ]
        )
        self.assertListEqual(self.network.get_interface_nodes(0), [("A", 0), ("B", 0)])
        self.assertListEqual(self.network.get_interface_nodes(1), [("B", 1), ("B", 1)])

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
            {cpd.variable for cpd in self.network.get_cpds()},
            {("I", 1), ("D", 0), ("G", 1), ("I", 0), ("G", 0), ("D", 1)},
        )
        self.assertEqual(
            {cpd.variable for cpd in self.network.get_cpds(time_slice=[0, 1])},
            {("I", 1), ("D", 0), ("G", 1), ("I", 0), ("G", 0), ("D", 1)},
        )
        self.assertEqual(
            set(self.network.get_cpds(time_slice=0)),
            set([self.diff_cpd, self.intel_cpd, self.grade_cpd]),
        )
        self.assertEqual(
            {cpd.variable for cpd in self.network.get_cpds(time_slice=1)},
            {("D", 1), ("I", 1), ("G", 1)},
        )
        self.assertRaises(ValueError, self.network.get_cpds, time_slice=-1)
        self.assertRaises(ValueError, self.network.get_cpds, time_slice=[0, 1.1])
        self.assertRaises(ValueError, self.network.get_cpds, time_slice="abc")

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
        self.assertIsInstance(copy, DBN)
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
        model = DBN(
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
        self.G = DBN()
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
        self.cancer_model = DBN()
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


class TestDBNSampling(unittest.TestCase):
    def setUp(self):
        self.dbn = DBN()
        self.dbn.add_edges_from(
            [
                (("D", 0), ("G", 0)),
                (("I", 0), ("G", 0)),
                (("D", 0), ("D", 1)),
                (("I", 0), ("I", 1)),
            ]
        )
        diff_cpd = TabularCPD(("D", 0), 2, [[0.6], [0.4]])
        grade_cpd = TabularCPD(
            ("G", 0),
            3,
            [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
            evidence=[("I", 0), ("D", 0)],
            evidence_card=[2, 2],
        )
        d_i_cpd = TabularCPD(
            ("D", 1),
            2,
            [[0.6, 0.3], [0.4, 0.7]],
            evidence=[("D", 0)],
            evidence_card=[2],
        )
        intel_cpd = TabularCPD(("I", 0), 2, [[0.7], [0.3]])
        i_i_cpd = TabularCPD(
            ("I", 1),
            2,
            [[0.5, 0.4], [0.5, 0.6]],
            evidence=[("I", 0)],
            evidence_card=[2],
        )
        g_i_cpd = TabularCPD(
            ("G", 1),
            3,
            [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
            evidence=[("I", 1), ("D", 1)],
            evidence_card=[2, 2],
        )
        self.dbn.add_cpds(diff_cpd, grade_cpd, d_i_cpd, intel_cpd, i_i_cpd, g_i_cpd)

        self.dbn_infer = DBNInference(self.dbn)

        # Construct an equivalent simple BN to match values to
        self.equivalent_bn = BayesianNetwork(
            [
                ("D0", "G0"),
                ("I0", "G0"),
                ("D0", "D1"),
                ("I0", "I1"),
                ("D1", "G1"),
                ("I1", "G1"),
                ("D1", "D2"),
                ("I1", "I2"),
                ("D2", "G2"),
                ("I2", "G2"),
            ]
        )
        d0_cpd = TabularCPD("D0", 2, [[0.6], [0.4]])
        i0_cpd = TabularCPD("I0", 2, [[0.7], [0.3]])
        g0_cpd = TabularCPD(
            "G0",
            3,
            [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
            evidence=["I0", "D0"],
            evidence_card=[2, 2],
        )
        d1_cpd = TabularCPD(
            "D1", 2, [[0.6, 0.3], [0.4, 0.7]], evidence=["D0"], evidence_card=[2]
        )
        i1_cpd = TabularCPD(
            "I1", 2, [[0.5, 0.4], [0.5, 0.6]], evidence=["I0"], evidence_card=[2]
        )
        g1_cpd = TabularCPD(
            "G1",
            3,
            [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
            evidence=["I1", "D1"],
            evidence_card=[2, 2],
        )
        d2_cpd = TabularCPD(
            "D2", 2, [[0.6, 0.3], [0.4, 0.7]], evidence=["D1"], evidence_card=[2]
        )
        i2_cpd = TabularCPD(
            "I2", 2, [[0.5, 0.4], [0.5, 0.6]], evidence=["I1"], evidence_card=[2]
        )
        g2_cpd = TabularCPD(
            "G2",
            3,
            [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
            evidence=["I2", "D2"],
            evidence_card=[2, 2],
        )
        self.equivalent_bn.add_cpds(
            d0_cpd, i0_cpd, g0_cpd, d1_cpd, i1_cpd, g1_cpd, d2_cpd, i2_cpd, g2_cpd
        )
        self.bn_infer = VariableElimination(self.equivalent_bn)
        self.bn_causal_infer = CausalInference(self.equivalent_bn)

    def test_simulate_two_slices(self):
        samples = self.dbn.simulate(n_samples=10, n_time_slices=1, show_progress=False)
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(samples.columns), 3)
        for node in [("D", 0), ("I", 0), ("G", 0)]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])

        samples = self.dbn.simulate(
            n_samples=int(1e5), n_time_slices=2, show_progress=False
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 6)
        for node in [("D", 0), ("I", 0), ("G", 0), ("D", 1), ("I", 1), ("G", 1)]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            dbn_infer_cpd = self.dbn_infer.query([node])[node]
            bn_infer_cpd = self.bn_infer.query([str(node[0]) + str(node[1])])
            for state in range(samples_cpd.shape[0]):
                self.assertTrue(
                    np.isclose(
                        sample_marginals[node].loc[state].values[0],
                        dbn_infer_cpd.values[state],
                        atol=0.01,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        sample_marginals[node].loc[state].values[0],
                        bn_infer_cpd.values[state],
                        atol=0.01,
                    )
                )

    def test_simulate_more_than_two_slices(self):
        samples = self.dbn.simulate(
            n_samples=int(1e5), n_time_slices=3, show_progress=False
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 9)
        for node in [
            ("D", 0),
            ("I", 0),
            ("G", 0),
            ("D", 1),
            ("I", 1),
            ("G", 1),
            ("D", 2),
            ("I", 2),
            ("G", 2),
        ]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 2)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            dbn_infer_cpd = self.dbn_infer.query([node])[node]
            bn_infer_cpd = self.bn_infer.query([str(node[0]) + str(node[1])])
            for state in range(samples_cpd.shape[0]):
                self.assertTrue(
                    np.isclose(
                        sample_marginals[node].loc[state].values[0],
                        dbn_infer_cpd.values[state],
                        atol=0.01,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        sample_marginals[node].loc[state].values[0],
                        bn_infer_cpd.values[state],
                        atol=0.01,
                    )
                )

    def test_simulate_evidence_two_slices(self):
        # Single evidence
        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=2,
            evidence={("D", 0): 1},
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 6)
        for node in [("D", 0), ("I", 0), ("G", 0), ("D", 1), ("I", 1), ("G", 1)]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            # DBN query only works for variables > evidence time
            if node[1] > 0:
                dbn_infer_cpd = self.dbn_infer.query([node], evidence={("D", 0): 1})[
                    node
                ]
            # Query can't have same node in variables and evidence
            if node != ("D", 0):
                bn_infer_cpd = self.bn_infer.query(
                    [str(node[0]) + str(node[1])], evidence={"D0": 1}
                )

            for state in range(samples_cpd.shape[0]):
                # TODO: DBN query with evidence values doesn't match with BN inference or sampling
                # if node[1] > 0:
                #     self.assertTrue(
                #         np.isclose(
                #             sample_marginals[node].loc[state].values[0],
                #             dbn_infer_cpd.values[state],
                #             atol=0.01,
                #         )
                #     )
                if node != ("D", 0):
                    self.assertTrue(
                        np.isclose(
                            sample_marginals[node].loc[state].values[0],
                            bn_infer_cpd.values[state],
                            atol=0.01,
                        )
                    )

        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=2,
            evidence={
                ("D", 0): 1,
                ("D", 1): 0,
            },
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 6)
        for node in [("D", 0), ("I", 0), ("G", 0), ("D", 1), ("I", 1), ("G", 1)]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            # Query can't have same node in variables and evidence
            if node not in [("D", 0), ("D", 1)]:
                bn_infer_cpd = self.bn_infer.query(
                    [str(node[0]) + str(node[1])], evidence={"D0": 1, "D1": 0}
                )

            for state in range(samples_cpd.shape[0]):
                if node not in [("D", 0), ("D", 1)]:
                    self.assertTrue(
                        np.isclose(
                            sample_marginals[node].loc[state].values[0],
                            bn_infer_cpd.values[state],
                            atol=0.01,
                        )
                    )

    def test_simulate_evidence_more_than_two_slices(self):
        # Evidence in first two slices
        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=3,
            evidence={
                ("D", 0): 1,
                ("D", 1): 0,
            },
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 9)
        for node in [
            ("D", 0),
            ("I", 0),
            ("G", 0),
            ("D", 1),
            ("I", 1),
            ("G", 1),
            ("D", 2),
            ("I", 2),
            ("G", 2),
        ]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 2)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            # Query can't have same node in variables and evidence
            if node not in [("D", 0), ("D", 1)]:
                bn_infer_cpd = self.bn_infer.query(
                    [str(node[0]) + str(node[1])], evidence={"D0": 1, "D1": 0}
                )

            for state in range(samples_cpd.shape[0]):
                if node not in [("D", 0), ("D", 1)]:
                    self.assertTrue(
                        np.isclose(
                            sample_marginals[node].loc[state].values[0],
                            bn_infer_cpd.values[state],
                            atol=0.01,
                        )
                    )

        # Evidence in third slices
        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=3,
            evidence={
                ("D", 0): 1,
                ("D", 1): 0,
                ("D", 2): 1,
            },
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 9)
        for node in [
            ("D", 0),
            ("I", 0),
            ("G", 0),
            ("D", 1),
            ("I", 1),
            ("G", 1),
            ("D", 2),
            ("I", 2),
            ("G", 2),
        ]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 2)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 2)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            # Query can't have same node in variables and evidence
            if node not in [("D", 0), ("D", 1), ("D", 2)]:
                bn_infer_cpd = self.bn_infer.query(
                    [str(node[0]) + str(node[1])], evidence={"D0": 1, "D1": 0, "D2": 1}
                )

            for state in range(samples_cpd.shape[0]):
                if node not in [("D", 0), ("D", 1), ("D", 2)]:
                    self.assertTrue(
                        np.isclose(
                            sample_marginals[node].loc[state].values[0],
                            bn_infer_cpd.values[state],
                            atol=0.01,
                        )
                    )

    def test_simulate_virtual_evidence(self):
        virtual_evidence = [
            TabularCPD(("D", 0), 2, [[0.2], [0.8]]),
            TabularCPD(("D", 2), 2, [[0.8], [0.2]]),
        ]
        bn_virtual_evidence = [
            TabularCPD("D0", 2, [[0.2], [0.8]]),
            TabularCPD("D2", 2, [[0.8], [0.2]]),
        ]
        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=3,
            virtual_evidence=virtual_evidence,
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 9)
        for node in [
            ("D", 0),
            ("I", 0),
            ("G", 0),
            ("D", 1),
            ("I", 1),
            ("G", 1),
            ("D", 2),
            ("I", 2),
            ("G", 2),
        ]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 2)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            bn_infer_cpd = self.bn_infer.query(
                [str(node[0]) + str(node[1])], virtual_evidence=bn_virtual_evidence
            )

            for state in range(samples_cpd.shape[0]):
                self.assertTrue(
                    np.isclose(
                        sample_marginals[node].loc[state].values[0],
                        bn_infer_cpd.values[state],
                        atol=0.05,
                    )
                )

    def test_simulate_intervention(self):
        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=3,
            do={("D", 0): 1, ("D", 2): 0},
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 9)
        for node in [
            ("D", 0),
            ("I", 0),
            ("G", 0),
            ("D", 1),
            ("I", 1),
            ("G", 1),
            ("D", 2),
            ("I", 2),
            ("G", 2),
        ]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 2)]].values)), [0])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 2)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            if node not in [("D", 0), ("D", 1), ("D", 2)]:
                bn_infer_cpd = self.bn_causal_infer.query(
                    [str(node[0]) + str(node[1])], do={"D0": 1, "D2": 0}
                )

            for state in range(samples_cpd.shape[0]):
                if node not in [("D", 0), ("D", 1), ("D", 2)]:
                    self.assertTrue(
                        np.isclose(
                            sample_marginals[node].loc[state].values[0],
                            bn_infer_cpd.values[state],
                            atol=0.08,
                        )
                    )

    def test_simulate_virtual_intervention(self):
        # Virtual intervention equivalent to hard intervention of (D, 0) = 1 and (D, 2) = 0
        virtual_intervention = [
            TabularCPD(("D", 0), 2, [[0], [1]]),
            TabularCPD(("D", 2), 2, [[1], [0]]),
        ]
        bn_virtual_intervention = [
            TabularCPD("D0", 2, [[0], [1]]),
            TabularCPD("D2", 2, [[1], [0]]),
        ]

        samples = self.dbn.simulate(
            n_samples=int(1e5),
            n_time_slices=3,
            virtual_intervention=virtual_intervention,
            show_progress=False,
        )
        self.assertEqual(len(samples), int(1e5))
        self.assertEqual(len(samples.columns), 9)
        for node in [
            ("D", 0),
            ("I", 0),
            ("G", 0),
            ("D", 1),
            ("I", 1),
            ("G", 1),
            ("D", 2),
            ("I", 2),
            ("G", 2),
        ]:
            self.assertIn(node, samples.columns)

        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 0)]].values)), [1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 0)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 0)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 1)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 1)]].values)), [0, 1, 2])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("D", 2)]].values)), [0])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("I", 2)]].values)), [0, 1])
        self.assertTrue(sorted(np.unique(samples.loc[:, [("G", 2)]].values)), [0, 1, 2])

        # Test the asymptotic distribution of samples
        sample_marginals = {
            node: samples.loc[:, [node.to_tuple()]].value_counts() / samples.shape[0]
            for node in self.dbn.nodes()
        }

        for node in sample_marginals.keys():
            samples_cpd = sample_marginals[node]
            if node not in [("D", 0), ("D", 1), ("D", 2)]:
                bn_infer_cpd = self.bn_causal_infer.query(
                    [str(node[0]) + str(node[1])], do={"D0": 1, "D2": 0}
                )

            for state in range(samples_cpd.shape[0]):
                if node not in [("D", 0), ("D", 1), ("D", 2)]:
                    self.assertTrue(
                        np.isclose(
                            sample_marginals[node].loc[state].values[0],
                            bn_infer_cpd.values[state],
                            atol=0.08,
                        )
                    )
