import unittest

import networkx as nx
import pandas as pd
import numpy as np
import numpy.testing as np_test

from pgmpy.models import BayesianModel, MarkovModel
import pgmpy.tests.help_functions as hf
from pgmpy.factors.discrete import TabularCPD, JointProbabilityDistribution, DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.estimators import BayesianEstimator, BaseEstimator, MaximumLikelihoodEstimator


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
        self.G1 = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        diff_cpd = TabularCPD('diff', 2, values=[[0.2], [0.8]])
        intel_cpd = TabularCPD('intel', 3, values=[[0.5], [0.3], [0.2]])
        grade_cpd = TabularCPD('grade', 3, values=[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                   [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                   [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                               evidence=['diff', 'intel'], evidence_card=[2, 3])
        self.G1.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        self.G2 = BayesianModel([('d', 'g'), ('g', 'l'), ('i', 'g'), ('i', 'l')])

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

    def test_get_ancestors_of_success(self):
        ancenstors1 = self.G2._get_ancestors_of('g')
        ancenstors2 = self.G2._get_ancestors_of('d')
        ancenstors3 = self.G2._get_ancestors_of(['i', 'l'])
        self.assertEqual(ancenstors1, {'d', 'i', 'g'})
        self.assertEqual(ancenstors2, {'d'})
        self.assertEqual(ancenstors3, {'g', 'i', 'l', 'd'})

    def test_get_ancestors_of_failure(self):
        self.assertRaises(ValueError, self.G2._get_ancestors_of, 'h')

    def test_get_cardinality(self):
        self.assertDictEqual(self.G1.get_cardinality(), {'diff': 2, 'intel': 3, 'grade': 3})

    def test_get_cardinality_with_node(self):
        self.assertEqual(self.G1.get_cardinality('diff'), 2)
        self.assertEqual(self.G1.get_cardinality('intel'), 3)
        self.assertEqual(self.G1.get_cardinality('grade'), 3)

    def test_local_independencies(self):
        self.assertEqual(self.G.local_independencies('a'), Independencies(['a', ['b', 'c']]))
        self.assertEqual(self.G.local_independencies('c'), Independencies(['c', ['a', 'd', 'e'], 'b']))
        self.assertEqual(self.G.local_independencies('d'), Independencies(['d', 'c', ['b', 'a']]))
        self.assertEqual(self.G.local_independencies('e'), Independencies(['e', ['c', 'b', 'a'], 'd']))
        self.assertEqual(self.G.local_independencies('b'), Independencies(['b', 'a']))
        self.assertEqual(self.G1.local_independencies('grade'), Independencies())

    def test_get_independencies(self):
        chain = BayesianModel([('X', 'Y'), ('Y', 'Z')])
        self.assertEqual(chain.get_independencies(), Independencies(('X', 'Z', 'Y'), ('Z', 'X', 'Y')))
        fork = BayesianModel([('Y', 'X'), ('Y', 'Z')])
        self.assertEqual(fork.get_independencies(), Independencies(('X', 'Z', 'Y'), ('Z', 'X', 'Y')))
        collider = BayesianModel([('X', 'Y'), ('Z', 'Y')])
        self.assertEqual(collider.get_independencies(), Independencies(('X', 'Z'), ('Z', 'X')))

    def test_is_imap(self):
        val = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
               0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]
        JPD = JointProbabilityDistribution(['diff', 'intel', 'grade'], [2, 3, 3], val)
        fac = DiscreteFactor(['diff', 'intel', 'grade'], [2, 3, 3], val)
        self.assertTrue(self.G1.is_imap(JPD))
        self.assertRaises(TypeError, self.G1.is_imap, fac)

    def test_get_immoralities(self):
        G = BayesianModel([('x', 'y'), ('z', 'y'), ('x', 'z'), ('w', 'y')])
        self.assertEqual(G.get_immoralities(), {('w', 'x'), ('w', 'z')})
        G1 = BayesianModel([('x', 'y'), ('z', 'y'), ('z', 'x'), ('w', 'y')])
        self.assertEqual(G1.get_immoralities(), {('w', 'x'), ('w', 'z')})
        G2 = BayesianModel([('x', 'y'), ('z', 'y'), ('x', 'z'), ('w', 'y'), ('w', 'x')])
        self.assertEqual(G2.get_immoralities(), {('w', 'z')})

    def test_is_iequivalent(self):
        G = BayesianModel([('x', 'y'), ('z', 'y'), ('x', 'z'), ('w', 'y')])
        self.assertRaises(TypeError, G.is_iequivalent, MarkovModel())
        G1 = BayesianModel([('V', 'W'), ('W', 'X'), ('X', 'Y'), ('Z', 'Y')])
        G2 = BayesianModel([('W', 'V'), ('X', 'W'), ('X', 'Y'), ('Z', 'Y')])
        self.assertTrue(G1.is_iequivalent(G2))
        G3 = BayesianModel([('W', 'V'), ('W', 'X'), ('Y', 'X'), ('Z', 'Y')])
        self.assertFalse(G3.is_iequivalent(G2))

    def test_copy(self):
        model_copy = self.G1.copy()
        self.assertEqual(sorted(self.G1.nodes()), sorted(model_copy.nodes()))
        self.assertEqual(sorted(self.G1.edges()), sorted(model_copy.edges()))
        self.assertNotEqual(id(self.G1.get_cpds('diff')),
                            id(model_copy.get_cpds('diff')))

        self.G1.remove_cpds('diff')
        diff_cpd = TabularCPD('diff', 2, values=[[0.3], [0.7]])
        self.G1.add_cpds(diff_cpd)
        self.assertNotEqual(self.G1.get_cpds('diff'),
                            model_copy.get_cpds('diff'))

        self.G1.remove_node('intel')
        self.assertNotEqual(sorted(self.G1.nodes()), sorted(model_copy.nodes()))
        self.assertNotEqual(sorted(self.G1.edges()), sorted(model_copy.edges()))

    def test_remove_node(self):
        self.G1.remove_node('diff')
        self.assertEqual(sorted(self.G1.nodes()), sorted(['grade', 'intel']))
        self.assertRaises(ValueError, self.G1.get_cpds, 'diff')

    def test_remove_nodes_from(self):
        self.G1.remove_nodes_from(['diff', 'grade'])
        self.assertEqual(sorted(self.G1.nodes()), sorted(['intel']))
        self.assertRaises(ValueError, self.G1.get_cpds, 'diff')
        self.assertRaises(ValueError, self.G1.get_cpds, 'grade')

    def tearDown(self):
        del self.G
        del self.G1


class TestBayesianModelCPD(unittest.TestCase):

    def setUp(self):
        self.G = BayesianModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
                                ('i', 's')])

    def test_active_trail_nodes(self):
        self.assertEqual(sorted(self.G.active_trail_nodes('d')['d']), ['d', 'g', 'l'])
        self.assertEqual(sorted(self.G.active_trail_nodes('i')['i']), ['g', 'i', 'l', 's'])
        self.assertEqual(sorted(self.G.active_trail_nodes(['d', 'i'])['d']), ['d', 'g', 'l'])

    def test_active_trail_nodes_args(self):
        self.assertEqual(sorted(self.G.active_trail_nodes(['d', 'l'], observed='g')['d']), ['d', 'i', 's'])
        self.assertEqual(sorted(self.G.active_trail_nodes(['d', 'l'], observed='g')['l']), ['l'])
        self.assertEqual(sorted(self.G.active_trail_nodes('s', observed=['i', 'l'])['s']), ['s'])
        self.assertEqual(sorted(self.G.active_trail_nodes('s', observed=['d', 'l'])['s']), ['g', 'i', 's'])

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
        cpd_d = TabularCPD('d', 2, values=np.random.rand(2, 1))
        cpd_i = TabularCPD('i', 2, values=np.random.rand(2, 1))
        cpd_g = TabularCPD('g', 2, values=np.random.rand(2, 4),
                           evidence=['d', 'i'], evidence_card=[2, 2])
        cpd_l = TabularCPD('l', 2, values=np.random.rand(2, 2),
                           evidence=['g'], evidence_card=[2])
        cpd_s = TabularCPD('s', 2, values=np.random.rand(2, 2),
                           evidence=['i'], evidence_card=[2])
        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

        self.assertEqual(self.G.get_cpds('d').variable, 'd')

    def test_get_cpds1(self):
        self.model = BayesianModel([('A', 'AB')])
        cpd_a = TabularCPD('A', 2, values=np.random.rand(2, 1))
        cpd_ab = TabularCPD('AB', 2, values=np.random.rand(2, 2),
                            evidence=['A'], evidence_card=[2])

        self.model.add_cpds(cpd_a, cpd_ab)
        self.assertEqual(self.model.get_cpds('A').variable, 'A')
        self.assertEqual(self.model.get_cpds('AB').variable, 'AB')
        self.assertRaises(ValueError, self.model.get_cpds, 'B')

        self.model.add_node('B')
        self.assertIsNone(self.model.get_cpds('B'))

    def test_add_single_cpd(self):
        cpd_s = TabularCPD('s', 2, np.random.rand(2, 2), ['i'], [2])
        self.G.add_cpds(cpd_s)
        self.assertListEqual(self.G.get_cpds(), [cpd_s])

    def test_add_multiple_cpds(self):
        cpd_d = TabularCPD('d', 2, values=np.random.rand(2, 1))
        cpd_i = TabularCPD('i', 2, values=np.random.rand(2, 1))
        cpd_g = TabularCPD('g', 2, values=np.random.rand(2, 4),
                           evidence=['d', 'i'], evidence_card=[2, 2])
        cpd_l = TabularCPD('l', 2, values=np.random.rand(2, 2),
                           evidence=['g'], evidence_card=[2])
        cpd_s = TabularCPD('s', 2, values=np.random.rand(2, 2),
                           evidence=['i'], evidence_card=[2])

        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
        self.assertEqual(self.G.get_cpds('d'), cpd_d)
        self.assertEqual(self.G.get_cpds('i'), cpd_i)
        self.assertEqual(self.G.get_cpds('g'), cpd_g)
        self.assertEqual(self.G.get_cpds('l'), cpd_l)
        self.assertEqual(self.G.get_cpds('s'), cpd_s)

    def test_check_model(self):
        cpd_g = TabularCPD('g', 2, values=np.array([[0.2, 0.3, 0.4, 0.6],
                                                    [0.8, 0.7, 0.6, 0.4]]),
                           evidence=['d', 'i'], evidence_card=[2, 2])

        cpd_s = TabularCPD('s', 2, values=np.array([[0.2, 0.3],
                                                    [0.8, 0.7]]),
                           evidence=['i'], evidence_card=[2])

        cpd_l = TabularCPD('l', 2, values=np.array([[0.2, 0.3],
                                                    [0.8, 0.7]]),
                           evidence=['g'], evidence_card=[2])

        self.G.add_cpds(cpd_g, cpd_s, cpd_l)
        self.assertRaises(ValueError, self.G.check_model)

        cpd_d = TabularCPD('d', 2, values=[[0.8, 0.2]])
        cpd_i = TabularCPD('i', 2, values=[[0.7, 0.3]])
        self.G.add_cpds(cpd_d, cpd_i)

        self.assertTrue(self.G.check_model())

    def test_check_model1(self):
        cpd_g = TabularCPD('g', 2, values=np.array([[0.2, 0.3],
                                                    [0.8, 0.7]]),
                           evidence=['i'], evidence_card=[2])
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_g = TabularCPD('g', 2, values=np.array([[0.2, 0.3, 0.4, 0.6],
                                                    [0.8, 0.7, 0.6, 0.4]]),
                           evidence=['d', 's'], evidence_card=[2, 2])
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_g = TabularCPD('g', 2, values=np.array([[0.2, 0.3],
                                                    [0.8, 0.7]]),
                           evidence=['l'], evidence_card=[2])
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_l = TabularCPD('l', 2, values=np.array([[0.2, 0.3],
                                                    [0.8, 0.7]]),
                           evidence=['d'], evidence_card=[2])
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_l)

        cpd_l = TabularCPD('l', 2, values=np.array([[0.2, 0.3, 0.4, 0.6],
                                                    [0.8, 0.7, 0.6, 0.4]]),
                           evidence=['d', 'i'], evidence_card=[2, 2])
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_l)

        cpd_l = TabularCPD('l', 2, values=np.array([[0.2, 0.3, 0.4, 0.6, 0.2, 0.3, 0.4, 0.6],
                                                    [0.8, 0.7, 0.6, 0.4, 0.8, 0.7, 0.6, 0.4]]),
                           evidence=['g', 'd', 'i'], evidence_card=[2, 2, 2])
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_l)

    def test_check_model2(self):
        cpd_s = TabularCPD('s', 2, values=np.array([[0.5, 0.3],
                                                    [0.8, 0.7]]),
                           evidence=['i'], evidence_card=[2])
        self.G.add_cpds(cpd_s)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_s)

        cpd_g = TabularCPD('g', 2, values=np.array([[0.2, 0.3, 0.4, 0.6],
                                                    [0.3, 0.7, 0.6, 0.4]]),
                           evidence=['d', 'i'], evidence_card=[2, 2])
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_l = TabularCPD('l', 2, values=np.array([[0.2, 0.3],
                                                    [0.1, 0.7]]),
                           evidence=['g'], evidence_card=[2])
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_l)

    def tearDown(self):
        del self.G


class TestBayesianModelFitPredict(unittest.TestCase):

    def setUp(self):
        self.model_disconnected = BayesianModel()
        self.model_disconnected.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
        self.model_connected = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])

        self.model2 = BayesianModel([('A', 'C'), ('B', 'C')])
        self.data1 = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        self.data2 = pd.DataFrame(data={'A': [0, np.NaN, 1],
                                        'B': [0, 1, 0],
                                        'C': [1, 1, np.NaN],
                                        'D': [np.NaN, 'Y', np.NaN]})

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/titanic_train.csv', dtype=str)
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_bayesian_fit(self):
        print(isinstance(BayesianEstimator, BaseEstimator))
        print(isinstance(MaximumLikelihoodEstimator, BaseEstimator))
        self.model2.fit(self.data1, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=[9, 3])
        self.assertEqual(self.model2.get_cpds('B'), TabularCPD('B', 2, [[11.0 / 15], [4.0 / 15]]))

    def test_fit_missing_data(self):
        self.model2.fit(self.data2, state_names={'C': [0, 1]}, complete_samples_only=False)
        cpds = set([TabularCPD('A', 2, [[0.5], [0.5]]),
                    TabularCPD('B', 2, [[2. / 3], [1. / 3]]),
                    TabularCPD('C', 2, [[0, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5]],
                               evidence=['A', 'B'], evidence_card=[2, 2])])
        self.assertSetEqual(cpds, set(self.model2.get_cpds()))

    def test_disconnected_fit(self):
        values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
                              columns=['A', 'B', 'C', 'D', 'E'])
        self.model_disconnected.fit(values)

        for node in ['A', 'B', 'C', 'D', 'E']:
            cpd = self.model_disconnected.get_cpds(node)
            self.assertEqual(cpd.variable, node)
            np_test.assert_array_equal(cpd.cardinality, np.array([2]))
            value = (values.ix[:, node].value_counts() /
                     values.ix[:, node].value_counts().sum())
            value = value.reindex(sorted(value.index)).values
            np_test.assert_array_equal(cpd.values, value)

    def test_predict(self):
        titanic = BayesianModel()
        titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived")])
        titanic.fit(self.titanic_data2[500:])

        p1 = titanic.predict(self.titanic_data2[["Sex", "Pclass"]][:30])
        p2 = titanic.predict(self.titanic_data2[["Survived", "Pclass"]][:30])
        p3 = titanic.predict(self.titanic_data2[["Survived", "Sex"]][:30])

        p1_res = np.array(['0', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0', '1', '0',
                           '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                           '0', '0', '0', '0'])
        p2_res = np.array(['male', 'female', 'female', 'female', 'male', 'male', 'male',
                           'male', 'female', 'female', 'female', 'female', 'male', 'male',
                           'male', 'female', 'male', 'female', 'male', 'female', 'male',
                           'female', 'female', 'female', 'male', 'female', 'male', 'male',
                           'female', 'male'])
        p3_res = np.array(['3', '1', '1', '1', '3', '3', '3', '3', '1', '1', '1', '1', '3',
                           '3', '3', '1', '3', '1', '3', '1', '3', '1', '1', '1', '3', '1',
                           '3', '3', '1', '3'])

        np_test.assert_array_equal(p1.values.ravel(), p1_res)
        np_test.assert_array_equal(p2.values.ravel(), p2_res)
        np_test.assert_array_equal(p3.values.ravel(), p3_res)

    def test_connected_predict(self):
        np.random.seed(42)
        values = pd.DataFrame(np.array(np.random.randint(low=0, high=2, size=(1000, 5)),
                                       dtype=str),
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
                                             1, 1, 1, 0], dtype=str))

    def test_connected_predict_probability(self):
        np.random.seed(42)
        values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, 5)),
                              columns=['A', 'B', 'C', 'D', 'E'])
        fit_data = values[:80]
        predict_data = values[80:].copy()
        self.model_connected.fit(fit_data)
        predict_data.drop('E', axis=1, inplace=True)
        e_prob = self.model_connected.predict_probability(predict_data)
        np_test.assert_allclose(e_prob.values.ravel(),
                                np.array([0.57894737, 0.42105263, 0.57894737, 0.42105263, 0.57894737,
                                          0.42105263, 0.5, 0.5, 0.57894737, 0.42105263,
                                          0.5, 0.5, 0.57894737, 0.42105263, 0.57894737,
                                          0.42105263, 0.57894737, 0.42105263, 0.5, 0.5,
                                          0.57894737, 0.42105263, 0.57894737, 0.42105263, 0.5,
                                          0.5, 0.57894737, 0.42105263, 0.57894737, 0.42105263,
                                          0.5, 0.5, 0.57894737, 0.42105263, 0.5,
                                          0.5, 0.5, 0.5, 0.5, 0.5]), atol=0)
        predict_data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1, 5)),
                                    columns=['A', 'B', 'C', 'F', 'E'])[:]

    def test_predict_probability_errors(self):
        np.random.seed(42)
        values = pd.DataFrame(np.random.randint(low=0, high=2, size=(2, 5)),
                              columns=['A', 'B', 'C', 'D', 'E'])
        fit_data = values[:1]
        predict_data = values[1:].copy()
        self.model_connected.fit(fit_data)
        self.assertRaises(ValueError, self.model_connected.predict_probability, predict_data)
        predict_data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1, 5)),
                                    columns=['A', 'B', 'C', 'F', 'E'])[:]
        self.assertRaises(ValueError, self.model_connected.predict_probability, predict_data)

    def tearDown(self):
        del self.model_connected
        del self.model_disconnected


class TestDirectedGraphCPDOperations(unittest.TestCase):

    def setUp(self):
        self.graph = BayesianModel()

    def test_add_single_cpd(self):
        cpd = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                         evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd)
        self.assertListEqual(self.graph.get_cpds(), [cpd])

    def test_add_multiple_cpds(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.assertListEqual(self.graph.get_cpds(), [cpd1, cpd2, cpd3])

    def test_remove_single_cpd(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds(cpd1)
        self.assertListEqual(self.graph.get_cpds(), [cpd2, cpd3])

    def test_remove_multiple_cpds(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds(cpd1, cpd3)
        self.assertListEqual(self.graph.get_cpds(), [cpd2])

    def test_remove_single_cpd_string(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds('diff')
        self.assertListEqual(self.graph.get_cpds(), [cpd2, cpd3])

    def test_remove_multiple_cpds_string(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds('diff', 'grade')
        self.assertListEqual(self.graph.get_cpds(), [cpd2])

    def test_get_values_for_node(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.assertEqual(self.graph.get_cpds('diff'), cpd1)
        self.assertEqual(self.graph.get_cpds('intel'), cpd2)
        self.assertEqual(self.graph.get_cpds('grade'), cpd3)

    def test_get_values_raises_error(self):
        cpd1 = TabularCPD('diff', 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD('intel', 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD('grade', 2, values=np.random.rand(2, 4),
                          evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.graph.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.assertRaises(ValueError, self.graph.get_cpds, 'sat')

    def tearDown(self):
        del self.graph
