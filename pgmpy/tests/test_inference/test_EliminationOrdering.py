import unittest

import numpy as np

from pgmpy.inference import WeightedMinFill, MinNeighbours, MinWeight, MinFill
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD


class TestEliminationOrdering(unittest.TestCase):

    def setUp(self):
        self.model = BayesianModel()
        self.model.add_edges_from([('c', 'd'), ('d', 'g'), ('i', 'g'),
                              ('i', 's'), ('s', 'j'), ('g', 'l'),
                              ('l', 'j'), ('j', 'h'), ('g', 'h')])
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
        self.model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j,
                            cpd_l, cpd_h)

    def test_weighted_min_fill(self):
        elim_order = WeightedMinFill(self.model)
        self.assertEqual(elim_order.cost('c'), 0)
        self.assertEqual(elim_order.cost('d'), 10)

    def test_min_neighbors(self):
        elim_order = MinNeighbours(self.model)
        self.assertEqual(elim_order.cost('c'), 1)
        self.assertEqual(elim_order.cost('g'), 5)

    def test_min_weight(self):
        elim_order = MinWeight(self.model)
        self.assertEqual(elim_order.cost('l'), 12)
        self.assertEqual(elim_order.cost('s'), 8)

    def test_min_fill(self):
        elim_order = MinFill(self.model)
        self.assertEqual(elim_order.cost('i'), 2)
        self.assertEqual(elim_order.cost('h'), 0)

    def test_find_elimination_ordering(self):
        weighted_min_fill = WeightedMinFill(self.model)
        self.assertListEqual(weighted_min_fill.get_elimination_order(['c', 'd', 'g', 'l', 's']),
                                                                     ['c', 'l', 's', 'd', 'g'])

        min_neighbors = MinNeighbours(self.model)
        self.assertListEqual(min_neighbors.get_elimination_order(['c', 'd', 'g', 'l', 's']),
                                                                 ['c', 'd', 'l', 's', 'g'])

        min_weight = MinWeight(self.model)
        self.assertListEqual(min_weight.get_elimination_order(['c', 'd', 'g', 'l', 's']),
                                                              ['c', 's', 'd', 'l', 'g'])

        min_fill = MinFill(self.model)
        self.assertListEqual(min_fill.get_elimination_order(['c', 'd', 'g', 'l', 's']),
                                                            ['c', 'l', 'd', 's', 'g'])
