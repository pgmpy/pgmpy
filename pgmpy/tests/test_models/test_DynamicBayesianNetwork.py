import unittest

import pgmpy.tests.help_functions as hf
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np


class TestDynamicBayesianNetworkCreation(unittest.TestCase):

    def setUp(self):
        self.network = DynamicBayesianNetwork()

    def test_add_single_node(self):
        self.network.add_node('a')
        self.assertListEqual(self.network.nodes(), ['a'])

    def test_add_multiple_nodes(self):
        self.network.add_nodes_from(['a', 'b', 'c'])
        self.assertListEqual(sorted(self.network.nodes()), ['a', 'b', 'c'])

    def test_add_single_edge_with_timeslice(self):
        self.network.add_edge(('a', 0), ('b', 0))
        self.assertListEqual(sorted(self.network.edges()), [(('a', 0), ('b', 0)), (('a', 1), ('b', 1))])
        self.assertListEqual(sorted(self.network.nodes()), ['a', 'b'])

    def test_add_edge_with_different_number_timeslice(self):
        self.network.add_edge(('a', 2), ('b', 2))
        self.assertListEqual(sorted(self.network.edges()), [(('a', 0), ('b', 0)), (('a', 1), ('b', 1))])

    def test_add_edge_going_backward(self):
        self.assertRaises(NotImplementedError, self.network.add_edge, ('a', 1), ('b', 0))

    def test_add_edge_with_farther_timeslice(self):
        self.assertRaises(ValueError, self.network.add_edge, ('a', 2), ('b', 4))

    def test_add_edge_with_self_loop(self):
        self.assertRaises(ValueError, self.network.add_edge, ('a', 0), ('a', 0))

    def test_add_edge_with_varying_length(self):
        self.assertRaises(ValueError, self.network.add_edge, ('a', 1, 1), ('b', 2))
        self.assertRaises(ValueError, self.network.add_edge, ('b', 2), ('a', 2, 3))

    def test_add_edge_with_closed_path(self):
        self.assertRaises(ValueError, self.network.add_edges_from,
                          [(('a', 0), ('b', 0)), (('b', 0), ('c', 0)), (('c', 0), ('a', 0))])

    def test_add_single_edge_without_timeslice(self):
        self.assertRaises(ValueError, self.network.add_edge, 'a', 'b')

    def test_add_single_edge_with_incorrect_timeslice(self):
        self.assertRaises(ValueError, self.network.add_edge, ('a', 'b'), ('b', 'c'))

    def test_add_multiple_edges(self):
        self.network.add_edges_from([(('a', 0), ('b', 0)), (('a', 0), ('a', 1)), (('b', 0), ('b', 1))])
        self.assertListEqual(sorted(self.network.edges()),
                             [(('a', 0), ('a', 1)), (('a', 0), ('b', 0)), (('a', 1), ('b', 1)), (('b', 0), ('b', 1))])

    def tearDown(self):
        del self.network


class TestDynamicBayesianNetworkMethods(unittest.TestCase):

    def setUp(self):
        self.network = DynamicBayesianNetwork()
        self.grade_cpd = TabularCPD(('G', 0), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                                         [0.4, 0.25, 0.1, 0.3],
                                                         [0.3, 0.7, 0.1, 0.2]],
                                    evidence=[('D', 0), ('I', 0)], evidence_card=[2, 2])
        self.d_i_cpd = TabularCPD(('D', 1), 2, values=[[0.6, 0.3], [0.4, 0.7]],
                                  evidence=[('D', 0)], evidence_card=[2])
        self.diff_cpd = TabularCPD(('D', 0), 2, values=[[0.6, 0.4]])
        self.intel_cpd = TabularCPD(('I', 0), 2, values=[[0.7, 0.3]])
        self.i_i_cpd = TabularCPD(('I', 1), 2, values=[[0.5, 0.4], [0.5, 0.6]],
                                  evidence=[('I', 0)], evidence_card=[2])
        self.grade_1_cpd = TabularCPD(('G', 1), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                                           [0.4, 0.25, 0.1, 0.3],
                                                           [0.3, 0.7, 0.1, 0.2]],
                                      evidence=[('D', 1), ('I', 1)], evidence_card=[2, 2])

    def test_get_intra_and_inter_edges(self):
        self.network.add_edges_from([(('a', 0), ('b', 0)), (('a', 0), ('a', 1)), (('b', 0), ('b', 1))])
        self.assertListEqual(sorted(self.network.get_intra_edges()), [(('a', 0), ('b', 0))])
        self.assertListEqual(sorted(self.network.get_intra_edges(1)), [(('a', 1), ('b', 1))])
        self.assertRaises(ValueError, self.network.get_intra_edges, -1)
        self.assertRaises(ValueError, self.network.get_intra_edges, '-')
        self.assertListEqual(sorted(self.network.get_inter_edges()), [(('a', 0), ('a', 1)), (('b', 0), ('b', 1))])

    def test_get_interface_nodes(self):
        self.network.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)), (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        self.assertListEqual(sorted(self.network.get_interface_nodes()), [('D', 0), ('I', 0)])
        self.assertRaises(ValueError, self.network.get_interface_nodes, -1)
        self.assertRaises(ValueError, self.network.get_interface_nodes, '-')

    def test_get_slice_nodes(self):
        self.network.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)), (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        self.assertListEqual(sorted(self.network.get_slice_nodes()), [('D', 0), ('G', 0), ('I', 0)])
        self.assertListEqual(sorted(self.network.get_slice_nodes(1)), [('D', 1), ('G', 1), ('I', 1)])
        self.assertRaises(ValueError, self.network.get_slice_nodes, -1)
        self.assertRaises(ValueError, self.network.get_slice_nodes, '-')

    def test_add_single_cpds(self):
        self.network.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0))])
        self.network.add_cpds(self.grade_cpd)
        self.assertListEqual(self.network.get_cpds(), [self.grade_cpd])

    def test_get_cpds(self):
        self.network.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)), (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        self.network.add_cpds(self.grade_cpd, self.d_i_cpd, self.diff_cpd, self.intel_cpd, self.i_i_cpd)
        self.network.initialize_initial_state()
        self.assertEqual(set(self.network.get_cpds()), set([self.diff_cpd, self.intel_cpd, self.grade_cpd]))
        self.assertEqual(self.network.get_cpds(time_slice=1)[0].variable, ('G', 1))

    def test_add_multiple_cpds(self):
        self.network.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)), (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        self.network.add_cpds(self.grade_cpd, self.d_i_cpd, self.diff_cpd, self.intel_cpd, self.i_i_cpd)
        self.assertEqual(self.network.get_cpds(('G', 0)).variable, ('G', 0))
        self.assertEqual(self.network.get_cpds(('D', 1)).variable, ('D', 1))
        self.assertEqual(self.network.get_cpds(('D', 0)).variable, ('D', 0))
        self.assertEqual(self.network.get_cpds(('I', 0)).variable, ('I', 0))
        self.assertEqual(self.network.get_cpds(('I', 1)).variable, ('I', 1))

    def test_initialize_initial_state(self):

        self.network.add_nodes_from(['D', 'G', 'I', 'S', 'L'])
        self.network.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)), (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        self.network.add_cpds(self.grade_cpd, self.d_i_cpd, self.diff_cpd, self.intel_cpd, self.i_i_cpd)
        self.network.initialize_initial_state()
        self.assertEqual(len(self.network.cpds), 6)
        self.assertEqual(self.network.get_cpds(('G', 1)).variable, ('G', 1))

    def test_moralize(self):
        self.network.add_edges_from(([(('D', 0), ('G', 0)), (('I', 0), ('G', 0))]))
        moral_graph = self.network.moralize()
        self.assertListEqual(hf.recursive_sorted(moral_graph.edges()),
                             [[('D', 0), ('G', 0)], [('D', 0), ('I', 0)],
                              [('D', 1), ('G', 1)], [('D', 1), ('I', 1)],
                              [('G', 0), ('I', 0)], [('G', 1), ('I', 1)]])

    def test_copy(self):
        self.network.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)), (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        cpd = TabularCPD(('G', 0), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                              [0.4, 0.25, 0.1, 0.3],
                                              [0.3, 0.7, 0.1, 0.2]],
                         evidence=[('D', 0), ('I', 0)], evidence_card=[2, 2])
        self.network.add_cpds(cpd)
        copy = self.network.copy()
        self.assertIsInstance(copy, DynamicBayesianNetwork)
        self.assertListEqual(sorted(self.network.nodes()), sorted(copy.nodes()))
        self.assertListEqual(sorted(self.network.edges()), sorted(copy.edges()))
        self.assertListEqual(self.network.get_cpds(), copy.get_cpds())
        self.assertListEqual(sorted(self.network.get_intra_edges()), sorted(copy.get_intra_edges()))
        self.assertListEqual(sorted(self.network.get_inter_edges()), sorted(copy.get_inter_edges()))
        self.assertListEqual(sorted(self.network.get_slice_nodes()), sorted(copy.get_slice_nodes()))

        copy.cpds[0].values = np.array([[0.4, 0.05, 0.3, 0.5], [0.3, 0.25, 0.5, 0.3], [0.3, 0.7, 0.2, 0.2]])
        self.assertNotEqual(self.network.get_cpds(), copy.get_cpds())
        self.network.add_cpds(self.i_i_cpd, self.d_i_cpd)

        copy.add_cpds(self.diff_cpd, self.intel_cpd)
        self.network.add_node('A')
        copy.add_node('Z')
        self.network.add_edge(('A', 0), ('D', 0))
        copy.add_edge(('Z', 0), ('D', 0))
        self.assertNotEqual(sorted(self.network.nodes()), sorted(copy.nodes()))
        self.assertNotEqual(sorted(self.network.edges()), sorted(copy.edges()))
        self.assertNotEqual(self.network.get_cpds(), copy.get_cpds())
        self.assertNotEqual(sorted(self.network.get_intra_edges()), sorted(copy.get_intra_edges()))
        self.assertListEqual(sorted(self.network.get_inter_edges()), sorted(copy.get_inter_edges()))
        self.assertNotEqual(sorted(self.network.get_slice_nodes()), sorted(copy.get_slice_nodes()))

        self.network.add_edge(('A', 0), ('D', 1))
        copy.add_edge(('Z', 0), ('D', 1))
        self.assertNotEqual(sorted(self.network.get_inter_edges()), sorted(copy.get_inter_edges()))

    def tearDown(self):
        del self.network


class TestDynamicBayesianNetworkMethods2(unittest.TestCase):

    def setUp(self):
        self.G = DynamicBayesianNetwork()
        self.G.add_edges_from(
            [(('D', 0), ('G', 0)), (('I', 0), ('G', 0)),
             (('D', 0), ('D', 1)), (('I', 0), ('I', 1))])
        """
        G.edges()
        [(('I', 0), ('G', 0)), (('I', 0), ('I', 1)),
         (('D', 1), ('G', 1)), (('D', 0), ('G', 0)),
         (('D', 0), ('D', 1)), (('I', 1), ('G', 1))]

        """

    def test_check_model(self):

        grade_cpd = TabularCPD(('G', 0), 3, values=[[0.3, 0.05, 0.7, 0.5],
                                                    [0.4, 0.25, 0.1, 0.3],
                                                    [0.3, 0.7, 0.2, 0.2]],
                               evidence=[('D', 0), ('I', 0)], evidence_card=[2, 2])

        d_i_cpd = TabularCPD(('D', 1), 2, values=[[0.6, 0.3], [0.4, 0.7]],
                             evidence=[('D', 0)], evidence_card=[2])

        diff_cpd = TabularCPD(('D', 0), 2, values=[[0.6, 0.4]])

        intel_cpd = TabularCPD(('I', 0), 2, values=[[0.7, 0.3]])

        i_i_cpd = TabularCPD(('I', 1), 2, values=[[0.5, 0.4], [0.5, 0.6]],
                             evidence=[('I', 0)], evidence_card=[2])

        grade_1_cpd = TabularCPD(('G', 1), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                                      [0.4, 0.25, 0.1, 0.3],
                                                      [0.3, 0.7, 0.1, 0.2]],
                                 evidence=[('D', 1), ('I', 1)], evidence_card=[2, 2])

        self.G.add_cpds(grade_cpd, d_i_cpd, i_i_cpd)
        self.assertTrue(self.G.check_model())

        self.G.remove_cpds(grade_cpd, d_i_cpd, i_i_cpd)
        self.G.add_cpds(grade_1_cpd, diff_cpd, intel_cpd)
        self.assertTrue(self.G.check_model())

    def test_check_model1(self):

        diff_cpd = TabularCPD(('D', 0), 3, values=[[0.3, 0.05, 0.7, 0.5],
                                                   [0.4, 0.25, 0.1, 0.3],
                                                   [0.3, 0.7, 0.2, 0.2]],
                              evidence=[('G', 0), ('I', 0)], evidence_card=[2, 2])
        self.G.add_cpds(diff_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(diff_cpd)

        grade_cpd = TabularCPD(('G', 0), 2, values=[[0.6, 0.3], [0.4, 0.7]],
                               evidence=[('D', 0)], evidence_card=[2])
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        diff_cpd = TabularCPD(('D', 0), 2, values=[[0.6, 0.3], [0.4, 0.7]],
                              evidence=[('D', 1)], evidence_card=[2])
        self.G.add_cpds(diff_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(diff_cpd)

        grade_cpd = TabularCPD(('G', 0), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                                    [0.4, 0.25, 0.1, 0.3],
                                                    [0.3, 0.7, 0.1, 0.2]],
                               evidence=[('D', 1), ('I', 1)], evidence_card=[2, 2])
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        grade_cpd = TabularCPD(('G', 1), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                                    [0.4, 0.25, 0.1, 0.3],
                                                    [0.3, 0.7, 0.1, 0.2]],
                               evidence=[('D', 0), ('I', 0)], evidence_card=[2, 2])
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        grade_cpd = TabularCPD(('G', 0), 2, values=[[0.6, 0.3], [0.4, 0.7]],
                               evidence=[('D', 1)], evidence_card=[2])
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

    def test_check_model2(self):

        grade_cpd = TabularCPD(('G', 0), 3, values=[[0.9, 0.05, 0.7, 0.5],
                                                    [0.4, 0.25, 0.1, 0.3],
                                                    [0.3, 0.7, 0.2, 0.2]],
                               evidence=[('D', 0), ('I', 0)], evidence_card=[2, 2])
        self.G.add_cpds(grade_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_cpd)

        d_i_cpd = TabularCPD(('D', 1), 2, values=[[0.1, 0.3], [0.4, 0.7]],
                             evidence=[('D', 0)], evidence_card=[2])
        self.G.add_cpds(d_i_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(d_i_cpd)

        diff_cpd = TabularCPD(('D', 0), 2, values=[[0.7, 0.4]])
        self.G.add_cpds(diff_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(diff_cpd)

        intel_cpd = TabularCPD(('I', 0), 2, values=[[1.7, 0.3]])
        self.G.add_cpds(intel_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(intel_cpd)

        i_i_cpd = TabularCPD(('I', 1), 2, values=[[0.9, 0.4], [0.5, 0.6]],
                             evidence=[('I', 0)], evidence_card=[2])
        self.G.add_cpds(i_i_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(i_i_cpd)

        grade_1_cpd = TabularCPD(('G', 1), 3, values=[[0.3, 0.05, 0.8, 0.5],
                                                      [0.4, 0.5, 0.1, 0.3],
                                                      [0.3, 0.7, 0.1, 0.2]],
                                 evidence=[('D', 1), ('I', 1)], evidence_card=[2, 2])
        self.G.add_cpds(grade_1_cpd)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(grade_1_cpd)

    def tearDown(self):
        del self.G
