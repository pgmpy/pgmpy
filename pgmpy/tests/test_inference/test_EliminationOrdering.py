import unittest
from pgmpy.inference import EliminationOrdering
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD


class TestEliminationOrdering(unittest.TestCase):

    def setUp(self):
        graph = BayesianModel()
        cpd_c = TabularCPD('c', 2, np.random.rand(2, 1))
        cpd_d = TabularCPD('d', 2, np.random.rand(2, 2),
                           ['c'], [2])
        cpd_g = TabularCPD('g', 3, np.random.rand(3, 4),
                           ['d', 'i'], [2, 2])
        cpd_i = TabularCPD('i', 2, np.random.rand(2, 1))
        cpd_s = TabularCPD('s', 2, np.random.rand(2, 2),
                           ['i'], [2])
        cpd_j = TabularCPD('j', 2, np.random.rand(2, 4),
                           ['l', 's'], [2, 2])
        cpd_l = TabularCPD('l', 2, np.random.rand(2, 3),
                           ['g'], [3])
        cpd_h = TabularCPD('h', 2, np.random.rand(2, 6),
                           ['g', 'j'], [3, 2])
        graph.add_edges_from([('c', 'd'), ('d', 'g'), ('i', 'g'),
                              ('i', 's'), ('s', 'j'), ('g', 'l'),
                              ('l', 'j'), ('j', 'h'), ('g', 'h')])
        graph.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j,
                       cpd_l, cpd_h)

        self.elim_ord = EliminationOrdering(graph)

    def test_weighted_min_fill(self):
        self.assertEqual(self.elim_ord.weighted_min_fill('c'), 0)
        self.assertEqual(self.elim_ord.weighted_min_fill('d'), 10)

    def test_min_neighbors(self):
        self.assertEqual(self.elim_ord.min_neighbors('c'), 1)
        self.assertEqual(self.elim_ord.min_neighbors('g'), 5)

    def test_min_weight(self):
        self.assertEqual(self.elim_ord.min_weight('l'), 12)
        self.assertEqual(self.elim_ord.min_weight('s'), 8)

    def test_min_fill(self):
        self.assertEqual(self.elim_ord.min_fill('i'), 2)
        self.assertEqual(self.elim_ord.min_fill('h'), 0)

    def test_find_elimination_ordering(self):
        self.assertListEqual(self.elim_ord.
                             find_elimination_ordering(['c', 'd',
                                                        'g', 'l', 's'],
                                                       self.elim_ord.
                                                       weighted_min_fill),
                             ['c', 'l', 's', 'd', 'g'])
        self.assertListEqual(self.elim_ord.
                             find_elimination_ordering(['c', 'd',
                                                        'g', 'l', 's'],
                                                       self.elim_ord.
                                                       min_neighbors),
                             ['c', 'd', 'l', 's', 'g'])
        self.assertListEqual(self.elim_ord.
                             find_elimination_ordering(['c', 'd',
                                                        'g', 'l', 's'],
                                                       self.elim_ord.
                                                       min_weight),
                             ['c', 's', 'd', 'l', 'g'])
        self.assertListEqual(self.elim_ord.
                             find_elimination_ordering(['c', 'd',
                                                        'g', 'l', 's'],
                                                       self.elim_ord.
                                                       min_fill),
                             ['c', 'l', 'd', 's', 'g'])
