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

    def test_elimination_order(self):
        # Just checking if it returns all the nodes passed to it.
        self.assertEqual(set(self.elimination_order.get_elimination_order()),
                         set(['sat', 'reco', 'intel', 'diff', 'grade']))
        self.assertEqual(set(self.elimination_order.get_elimination_order(nodes=['diff', 'grade', 'intel'])),
                         set(['grade', 'diff', 'intel']))

    def test_fill_in_edges(self):
        self.assertEqual(list(self.elimination_order.fill_in_edges('diff')), [])


class TestWeightedMinFill(BaseEliminationTest):
    def setUp(self):
        super(TestWeightedMinFill, self).setUp()
        self.elimination_order = WeightedMinFill(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost('diff'), 4)
        self.assertEqual(self.elimination_order.cost('intel'), 12)
