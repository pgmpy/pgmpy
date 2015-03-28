import unittest
import networkx as nx
import pandas as pd
import numpy as np
import numpy.testing as np_test
from pgmpy.models import BayesianModel
import pgmpy.tests.help_functions as hf
from pgmpy.factors import TabularCPD


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.G = BayesianModel()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = BayesianModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(sorted(self.g.nodes()), ['a', 'b', 'c'])
        self.assertListEqual(hf.recursive_sorted(self.g.edges()),
                             [['a', 'b'], ['b', 'c']])

    def test_class_init_with_data_nonstring(self):
        BayesianModel([(1, 2), (2, 3)])

    def test_add_node_string(self):
        self.G.add_node('a')
        self.assertListEqual(self.G.nodes(), ['a'])

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
        self.assertListEqual(self.G.edges(), [('d', 'e')])
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
        self.g = BayesianModel([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.g.predecessors('a'), [])
        self.assertListEqual(self.g.predecessors('b'), ['a'])
        self.assertListEqual(self.g.predecessors('c'), ['b'])

    def test_update_node_parents(self):
        self.G.add_nodes_from(['a', 'b', 'c'])
        self.G.add_edges_from([('a', 'b'), ('b', 'c')])
        self.assertListEqual(self.G.predecessors('a'), [])
        self.assertListEqual(self.G.predecessors('b'), ['a'])
        self.assertListEqual(self.G.predecessors('c'), ['b'])

    def tearDown(self):
        del self.G


class TestBayesianModelMethods(unittest.TestCase):
    def setUp(self):
        self.G = BayesianModel([('a', 'd'), ('b', 'd'),
                                ('d', 'e'), ('b', 'c')])

    def test_moral_graph(self):
        moral_graph = self.G.moralize()
        self.assertListEqual(sorted(moral_graph.nodes()), ['a', 'b', 'c', 'd', 'e'])
        for edge in moral_graph.edges():
            self.assertTrue(edge in [('a', 'b'), ('a', 'd'), ('b', 'c'), ('d', 'b'), ('e', 'd')] or
                            (edge[1], edge[0]) in [('a', 'b'), ('a', 'd'), ('b', 'c'), ('d', 'b'), ('e', 'd')])

    def test_moral_graph_with_edge_present_over_parents(self):
        G = BayesianModel([('a', 'd'), ('d', 'e'), ('b', 'd'), ('b', 'c'), ('a', 'b')])
        moral_graph = G.moralize()
        self.assertListEqual(sorted(moral_graph.nodes()), ['a', 'b', 'c', 'd', 'e'])
        for edge in moral_graph.edges():
            self.assertTrue(edge in [('a', 'b'), ('c', 'b'), ('d', 'a'), ('d', 'b'), ('d', 'e')] or
                            (edge[1], edge[0]) in [('a', 'b'), ('c', 'b'), ('d', 'a'), ('d', 'b'), ('d', 'e')])

    def tearDown(self):
        del self.G


class TestBayesianModelCPD(unittest.TestCase):
    def setUp(self):
        self.G = BayesianModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
                                ('i', 's')])

    def test_active_trail_nodes(self):
        self.assertEqual(sorted(self.G.active_trail_nodes('d')), ['d', 'g', 'l'])
        self.assertEqual(sorted(self.G.active_trail_nodes('i')), ['g', 'i', 'l', 's'])

    def test_active_trail_nodes_args(self):
        self.assertEqual(sorted(self.G.active_trail_nodes('d', observed='g')), ['d', 'i', 's'])
        self.assertEqual(sorted(self.G.active_trail_nodes('l', observed='g')), ['l'])
        self.assertEqual(sorted(self.G.active_trail_nodes('s', observed=['i', 'l'])), ['s'])
        self.assertEqual(sorted(self.G.active_trail_nodes('s', observed=['d', 'l'])), ['g', 'i', 's'])

    def test_is_active_trail_triplets(self):
        self.assertTrue(self.G.is_active_trail('d', 'l'))
        self.assertTrue(self.G.is_active_trail('g', 's'))
        self.assertFalse(self.G.is_active_trail('d', 'i'))
        self.assertTrue(self.G.is_active_trail('d', 'i', observed='g'))
        self.assertFalse(self.G.is_active_trail('d', 'l', observed='g'))
        self.assertFalse(self.G.is_active_trail('i', 'l', observed='g'))
        self.assertTrue(self.G.is_active_trail('d', 'i', observed='l'))
        self.assertFalse(self.G.is_active_trail('g', 's', observed='i'))

    def test_is_active_trail(self):
        self.assertFalse(self.G.is_active_trail('d', 's'))
        self.assertTrue(self.G.is_active_trail('s', 'l'))
        self.assertTrue(self.G.is_active_trail('d', 's', observed='g'))
        self.assertFalse(self.G.is_active_trail('s', 'l', observed='g'))

    def test_is_active_trail_args(self):
        self.assertFalse(self.G.is_active_trail('s', 'l', 'i'))
        self.assertFalse(self.G.is_active_trail('s', 'l', 'g'))
        self.assertTrue(self.G.is_active_trail('d', 's', 'l'))
        self.assertFalse(self.G.is_active_trail('d', 's', ['i', 'l']))

    def test_get_cpds(self):
        cpd_d = TabularCPD('d', 2, np.random.rand(2, 1))
        cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
        cpd_g = TabularCPD('g', 2, np.random.rand(2, 4), ['d', 'i'], [2, 2])
        cpd_l = TabularCPD('l', 2, np.random.rand(2, 2), ['g'], 2)
        cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], 2)
        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

        self.assertEqual(self.G.get_cpds('d').variable, 'd')

    def test_get_cpds1(self):
        self.model = BayesianModel([('A', 'AB')])
        cpd_a = TabularCPD('A', 2, np.random.rand(2, 1))
        cpd_ab = TabularCPD('AB', 2, np.random.rand(2, 2), evidence=['A'],
                            evidence_card=[2])

        self.model.add_cpds(cpd_a, cpd_ab)
        self.assertEqual(self.model.get_cpds('A').variable, 'A')
        self.assertEqual(self.model.get_cpds('AB').variable, 'AB')

    def test_add_single_cpd(self):
        from pgmpy.factors import TabularCPD
        cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], 2)
        self.G.add_cpds(cpd_s)
        self.assertListEqual(self.G.get_cpds(), [cpd_s])

    def test_add_multiple_cpds(self):
        from pgmpy.factors import TabularCPD
        cpd_d = TabularCPD('d', 2, np.random.rand(2, 1))
        cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
        cpd_g = TabularCPD('g', 2, np.random.rand(2, 4), ['d', 'i'], [2, 2])
        cpd_l = TabularCPD('l', 2, np.random.rand(2, 2), ['g'], 2)
        cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], 2)

        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
        self.assertEqual(self.G.get_cpds('d'), cpd_d)
        self.assertEqual(self.G.get_cpds('i'), cpd_i)
        self.assertEqual(self.G.get_cpds('g'), cpd_g)
        self.assertEqual(self.G.get_cpds('l'), cpd_l)
        self.assertEqual(self.G.get_cpds('s'), cpd_s)

    def tearDown(self):
        del self.G


class TestBayesianModelFitPredict(unittest.TestCase):
    def setUp(self):
        self.model_disconnected = BayesianModel()
        self.model_disconnected.add_nodes_from(['A', 'B', 'C', 'D', 'E'])

        self.model_connected = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])

    def test_disconnected_fit(self):
        values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
                              columns=['A', 'B', 'C', 'D', 'E'])
        self.model_disconnected.fit(values)

        for node in ['A', 'B', 'C', 'D', 'E']:
            cpd = self.model_disconnected.get_cpds(node)
            self.assertEqual(cpd.variable, node)
            np_test.assert_array_equal(cpd.cardinality, np.array([2]))
            value = (values.ix[:, node].value_counts() /
                     values.ix[:, node].value_counts().sum()).values
            np_test.assert_array_equal(cpd.values, value)

    def test_connected_predict(self):
        np.random.seed(42)
        values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
                              columns=['A', 'B', 'C', 'D', 'E'])
        fit_data = values[:800]
        predict_data = values[800:].copy()
        self.model_connected.fit(fit_data)
        self.assertRaises(ValueError, self.model_connected.predict, predict_data)
        predict_data.drop('E', axis=1, inplace=True)
        e_predict = self.model_connected.predict(predict_data)
        np_test.assert_array_equal(e_predict.values.ravel(),
                                   np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                                             1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0,
                                             0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
                                             0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
                                             0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                                             1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
                                             1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                                             1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
                                             0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                                             1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                             1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1,
                                             0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                                             1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
                                             1, 1, 1, 0]))

    def tearDown(self):
        del self.model_connected
        del self.model_disconnected


class TestDirectedGraphCPDOperations(unittest.TestCase):
    def setUp(self):
        self.graph = BayesianModel()

    def test_add_single_cpd(self):
        cpd = TabularCPD('grade', 2, np.random.rand(2, 4),
                         ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd)
        self.assertListEqual(self.graph.get_cpds(), [cpd])

    def test_add_multiple_cpds(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.assertListEqual(self.graph.get_cpds(), [cpd1, cpd2, cpd3])

    def test_remove_single_cpd(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds(cpd1)
        self.assertListEqual(self.graph.get_cpds(), [cpd2, cpd3])

    def test_remove_multiple_cpds(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds(cpd1, cpd3)
        self.assertListEqual(self.graph.get_cpds(), [cpd2])

    def test_remove_single_cpd_string(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds('diff')
        self.assertListEqual(self.graph.get_cpds(), [cpd2, cpd3])

    def test_remove_multiple_cpds_string(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds('diff', 'grade')
        self.assertListEqual(self.graph.get_cpds(), [cpd2])

    def test_get_cpd_for_node(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.assertEqual(self.graph.get_cpds('diff'), cpd1)
        self.assertEqual(self.graph.get_cpds('intel'), cpd2)
        self.assertEqual(self.graph.get_cpds('grade'), cpd3)

    def test_get_cpd_raises_error(self):
        cpd1 = TabularCPD('diff', 2, np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, np.random.rand(2, 4),
                          ['diff', 'intel'], [2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.assertRaises(ValueError, self.graph.get_cpds, 'sat')

    def tearDown(self):
        del self.graph
