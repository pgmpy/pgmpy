import unittest
import pandas as pd
import numpy as np
import networkx as nx

from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.independencies import Independencies, IndependenceAssertion
from pgmpy.models import BayesianModel
from pgmpy.base import DirectedGraph, UndirectedGraph


class TestBaseEstimator(unittest.TestCase):
    def setUp(self):
        # link to dataset: "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/titanic_train.csv')

    def test_test_conditional_independence(self):
        data = pd.DataFrame(np.random.randint(0, 2, size=(1000, 4)), columns=list('ABCD'))
        data['E'] = data['A'] + data['B'] + data['C']
        est = ConstraintBasedEstimator(data)

        self.assertGreater(est.test_conditional_independence('A', 'C')[1], 0.01)  # independent
        self.assertGreater(est.test_conditional_independence('A', 'B', 'D')[1], 0.01)  # independent
        self.assertLess(est.test_conditional_independence('A', 'B', ['D', 'E'])[1], 0.01)  # dependent

    def test_test_conditional_independence_titanic(self):
        est = ConstraintBasedEstimator(self.titanic_data)

        self.assertEqual(est.test_conditional_independence('Embarked', 'Sex'),
                         (13.355630515001746, 0.020264556044311655, True))
        self.assertEqual(est.test_conditional_independence('Pclass', 'Survived', ['Embarked']),
                         (96.403283942888635, 4.1082315854166553e-13, True))
        self.assertEqual(est.test_conditional_independence('Embarked', 'Survived', ["Sex", "Pclass"]),
                         (21.537481934494085, 0.96380273702382602, True))
        # insufficient data test commented out, because generates warning
        # self.assertEqual(est.test_conditional_independence('Sex', 'Survived', ["Age", "Embarked"]),
        #                 (235.51133052530713, 0.99999999683394869, False))

    def test_build_skeleton(self):
        ind = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])
        ind = ind.closure()
        skel1, sep_sets1 = ConstraintBasedEstimator.build_skeleton("ABCD", ind)
        self.assertTrue(self._edge_list_equal(skel1.edges(), [('A', 'D'), ('B', 'D'), ('C', 'D')]))

        sep_sets_ref1 = {frozenset({'A', 'C'}): (), frozenset({'A', 'B'}): (), frozenset({'C', 'B'}): ()}
        self.assertEqual(sep_sets1, sep_sets_ref1)

        model = BayesianModel([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
        skel2, sep_sets2 = ConstraintBasedEstimator.build_skeleton(model.nodes(), model.get_independencies())
        self.assertTrue(self._edge_list_equal(skel2, [('D', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'E')]))

        sep_sets_ref2 = {frozenset({'D', 'C'}): ('B',),
                         frozenset({'E', 'B'}): ('C',),
                         frozenset({'A', 'D'}): (),
                         frozenset({'E', 'D'}): ('C',),
                         frozenset({'E', 'A'}): ('C',),
                         frozenset({'A', 'B'}): ()}
        # witnesses/seperators might change on each run, so we cannot compare directly
        self.assertEqual(sep_sets2.keys(), sep_sets_ref2.keys())
        self.assertEqual([len(v) for v in sorted(sep_sets2.values())],
                         [len(v) for v in sorted(sep_sets_ref2.values())])

    def test_skeleton_to_pdag(self):
        data = pd.DataFrame(np.random.randint(0, 3, size=(1000, 3)), columns=list('ABD'))
        data['C'] = data['A'] - data['B']
        data['D'] += data['A']
        c = ConstraintBasedEstimator(data)
        pdag = c.skeleton_to_pdag(*c.estimate_skeleton())
        self.assertSetEqual(set(pdag.edges()),
                            set([('B', 'C'), ('A', 'D'), ('A', 'C'), ('D', 'A')]))

        skel = UndirectedGraph([('A', 'B'), ('A', 'C')])
        sep_sets1 = {frozenset({'B', 'C'}): ()}
        self.assertSetEqual(set(c.skeleton_to_pdag(skel, sep_sets1).edges()),
                            set([('B', 'A'), ('C', 'A')]))

        sep_sets2 = {frozenset({'B', 'C'}): ('A',)}
        pdag2 = c.skeleton_to_pdag(skel, sep_sets2)
        self.assertSetEqual(set(c.skeleton_to_pdag(skel, sep_sets2).edges()),
                            set([('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A')]))

    def test_pdag_to_dag(self):
        pdag1 = DirectedGraph([('A', 'B'), ('C', 'B'), ('C', 'D'), ('D', 'C'), ('D', 'A'), ('A', 'D')])
        dag1 = ConstraintBasedEstimator.pdag_to_dag(pdag1)
        self.assertTrue(('A', 'B') in dag1.edges() and
                        ('C', 'B') in dag1.edges() and
                        len(dag1.edges()) == 4)

        pdag2 = DirectedGraph([('B', 'C'), ('D', 'A'), ('A', 'D'), ('A', 'C')])
        dag2 = ConstraintBasedEstimator.pdag_to_dag(pdag2)
        self.assertTrue(set(dag2.edges()) == set([('B', 'C'), ('A', 'D'), ('A', 'C')]) or
                        set(dag2.edges()) == set([('B', 'C'), ('D', 'A'), ('A', 'C')]))

        pdag3 = DirectedGraph([('B', 'C'), ('D', 'C'), ('C', 'D'), ('A', 'C')])
        dag3 = ConstraintBasedEstimator.pdag_to_dag(pdag3)
        self.assertSetEqual(set([('B', 'C'), ('C', 'D'), ('A', 'C')]),
                            set(dag3.edges()))

    def test_estimate_from_independencies(self):
        ind = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])
        ind = ind.closure()
        model = ConstraintBasedEstimator.estimate_from_independencies("ABCD", ind)

        self.assertSetEqual(set(model.edges()),
                            set([('B', 'D'), ('A', 'D'), ('C', 'D')]))

        model1 = BayesianModel([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
        model2 = ConstraintBasedEstimator.estimate_from_independencies(
                            model1.nodes(),
                            model1.get_independencies())

        self.assertTrue(set(model2.edges()) == set(model1.edges()) or
                        set(model2.edges()) == set([('B', 'C'), ('A', 'C'), ('C', 'E'), ('D', 'B')]))

    def test_estimate_skeleton(self):
        data = pd.DataFrame(np.random.randint(0, 2, size=(1000, 5)), columns=list('ABCDE'))
        data['F'] = data['A'] + data['B'] + data['C']
        est = ConstraintBasedEstimator(data)
        skel, sep_sets = est.estimate_skeleton()
        self.assertTrue(self._edge_list_equal(skel.edges(),
                                              [('A', 'F'), ('B', 'F'), ('C', 'F')]))

        sep_sets_ref = {frozenset({'D', 'F'}): (),
                        frozenset({'D', 'B'}): (),
                        frozenset({'A', 'C'}): (),
                        frozenset({'D', 'E'}): (),
                        frozenset({'E', 'F'}): (),
                        frozenset({'E', 'C'}): (),
                        frozenset({'E', 'B'}): (),
                        frozenset({'D', 'C'}): (),
                        frozenset({'A', 'B'}): (),
                        frozenset({'A', 'E'}): (),
                        frozenset({'B', 'C'}): (),
                        frozenset({'A', 'D'}): ()}
        self.assertEqual(set(sep_sets.keys()), set(sep_sets_ref.keys()))

    def test_estimate_skeleton2(self):
        data = pd.DataFrame(np.random.randint(0, 2, size=(1000, 3)), columns=list('XYZ'))
        data['X'] += data['Z']
        data['Y'] += data['Z']
        est = ConstraintBasedEstimator(data)
        skel, sep_sets = est.estimate_skeleton()

        self.assertTrue(self._edge_list_equal(skel.edges(), [('X', 'Z'), ('Y', 'Z')]))
        self.assertEqual(sep_sets, {frozenset(('X', 'Y')): ('Z',)})

    def test_estimate(self):
        data = pd.DataFrame(np.random.randint(0, 3, size=(1000, 3)), columns=list('XYZ'))
        data['sum'] = data.sum(axis=1)
        model = ConstraintBasedEstimator(data).estimate()
        self.assertSetEqual(set(model.edges()),
                            set([('Z', 'sum'), ('X', 'sum'), ('Y', 'sum')]))

    @staticmethod
    def _edge_list_equal(edges1, edges2):
        "Checks if two lists of undirected edges are equal."
        return ((((X, Y) in edges2 or (Y, X) in edges2) for X, Y in edges1) and
                (((X, Y) in edges1 or (Y, X) in edges1) for X, Y in edges2))

    def tearDown(self):
        del self.titanic_data
