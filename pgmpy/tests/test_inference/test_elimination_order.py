from unittest import TestCase

import numpy as np
import pandas as pd

from pgmpy.models import BayesianModel
from pgmpy.inference.EliminationOrder import (BaseEliminationOrder, WeightedMinFill,
                                              MinNeighbours, MinWeight, MinFill)


class BaseEliminationTest(TestCase):
    def setUp(self):
        self.model = BayesianModel([('diff', 'grade'), ('intel', 'grade'), ('intel', 'sat'),
                                    ('grade', 'reco')])
        raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
        data = pd.DataFrame(raw_data, columns=['diff', 'grade', 'intel', 'sat', 'reco'])
        self.model.fit(data)

    def tearDown(self):
        del self.model
        del self.elimination_order


class TestBaseElimination(BaseEliminationTest):
    def setUp(self):
        super(TestBaseElimination, self).setUp()
        self.elimination_order = BaseEliminationOrder(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost('grade'), 0)
        self.assertEqual(self.elimination_order.cost('diff'), 0)

    def test_fill_in_edges(self):
        self.assertEqual(list(self.elimination_order.fill_in_edges('diff')), [])


class TestWeightedMinFill(BaseEliminationTest):
    def setUp(self):
        super(TestWeightedMinFill, self).setUp()
        self.elimination_order = WeightedMinFill(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost('diff'), 4)
        self.assertEqual(self.elimination_order.cost('intel'), 12)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order()

        # Can't do a simple assertEqual because of the order of nodes having same score.
        self.assertTrue(all([elimination_order[0] in ['sat', 'reco'],
                             elimination_order[1] in ['sat', 'reco'],
                             elimination_order[2] in ['diff'],
                             elimination_order[3] in ['grade', 'intel'],
                             elimination_order[4] in ['grade', 'intel']]))

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=['diff', 'grade', 'sat'])
        self.assertEqual(elimination_order, ['sat', 'diff', 'grade'])


class TestMinNeighbours(BaseEliminationTest):
    def setUp(self):
        super(TestMinNeighbours, self).setUp()
        self.elimination_order = MinNeighbours(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost('grade'), 3)
        self.assertEqual(self.elimination_order.cost('reco'), 1)
        self.assertEqual(self.elimination_order.cost('intel'), 3)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order()
        self.assertTrue(all([elimination_order[0] in ['sat', 'reco'],
                             elimination_order[1] in ['sat', 'reco'],
                             elimination_order[2] in ['diff'],
                             elimination_order[3] in ['grade', 'intel'],
                             elimination_order[4] in ['grade', 'intel']]))

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=['diff', 'grade', 'sat'])
        self.assertEqual(elimination_order, ['sat', 'diff', 'grade'])


class TestMinWeight(BaseEliminationTest):
    def setUp(self):
        super(TestMinWeight, self).setUp()
        self.elimination_order = MinWeight(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost('diff'), 4)
        self.assertEqual(self.elimination_order.cost('intel'), 8)
        self.assertEqual(self.elimination_order.cost('reco'), 2)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order()
        self.assertTrue(all([elimination_order[0] in ['sat', 'reco'],
                             elimination_order[1] in ['sat', 'reco'],
                             elimination_order[2] in ['diff'],
                             elimination_order[3] in ['grade', 'intel'],
                             elimination_order[4] in ['grade', 'intel']]))

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=['diff', 'grade', 'sat'])
        self.assertEqual(elimination_order, ['sat', 'diff', 'grade'])


class TestMinFill(BaseEliminationTest):
    def setUp(self):
        super(TestMinFill, self).setUp()
        self.elimination_order = MinFill(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost('diff'), 0)
        self.assertEqual(self.elimination_order.cost('intel'), 1)
        self.assertEqual(self.elimination_order.cost('sat'), 0)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order()
        self.assertTrue(all([elimination_order[0] in ['diff', 'grade', 'sat', 'reco'],
                             elimination_order[1] in ['diff', 'grade', 'sat', 'reco'],
                             elimination_order[2] in ['diff', 'grade', 'sat', 'reco'],
                             elimination_order[3] in ['diff', 'grade', 'sat', 'reco'],
                             elimination_order[4] in ['intel']]))

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=['diff', 'grade', 'intel'])
        self.assertTrue(all([elimination_order[0] in ['diff', 'grade'],
                             elimination_order[1] in ['diff', 'grade'],
                             elimination_order[2] in ['intel']]))
