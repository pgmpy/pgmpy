from unittest import TestCase

import numpy as np
import pandas as pd

from pgmpy.inference.EliminationOrder import (
    BaseEliminationOrder,
    MinFill,
    MinNeighbors,
    MinWeight,
    WeightedMinFill,
)
from pgmpy.models import BayesianNetwork


class BaseEliminationTest(TestCase):
    def setUp(self):
        self.model = BayesianNetwork(
            [("diff", "grade"), ("intel", "grade"), ("intel", "sat"), ("grade", "reco")]
        )
        raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
        data = pd.DataFrame(raw_data, columns=["diff", "grade", "intel", "sat", "reco"])
        self.model.fit(data)

    def tearDown(self):
        del self.model
        del self.elimination_order


class TestBaseElimination(BaseEliminationTest):
    def setUp(self):
        super(TestBaseElimination, self).setUp()
        self.elimination_order = BaseEliminationOrder(self.model)

    def test_cost(self):
        costs = {"diff": 0, "sat": 0, "reco": 0, "grade": 0, "intel": 0}
        for var, expected_cost in costs.items():
            self.assertEqual(self.elimination_order.cost(var), expected_cost)

    def test_fill_in_edges(self):
        self.assertEqual(list(self.elimination_order.fill_in_edges("diff")), [])


class TestWeightedMinFill(BaseEliminationTest):
    def setUp(self):
        super(TestWeightedMinFill, self).setUp()
        self.elimination_order = WeightedMinFill(self.model)

    def test_cost(self):
        costs = {"diff": 4, "sat": 0, "reco": 0, "grade": 12, "intel": 12}
        for var, expected_cost in costs.items():
            self.assertEqual(self.elimination_order.cost(var), expected_cost)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        self.assertEqual(set(elimination_order[:2]), {"sat", "reco"})
        self.assertEqual(set(elimination_order[2:]), {"grade", "intel", "diff"})

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        self.assertEqual(elimination_order, ["sat", "diff", "grade"])


class TestMinNeighbors(BaseEliminationTest):
    def setUp(self):
        super(TestMinNeighbors, self).setUp()
        self.elimination_order = MinNeighbors(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost("grade"), 3)
        self.assertEqual(self.elimination_order.cost("reco"), 1)
        self.assertEqual(self.elimination_order.cost("intel"), 3)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        self.assertEqual(set(elimination_order[:2]), {"sat", "reco"})
        self.assertEqual(set(elimination_order[2:]), {"diff", "grade", "intel"})

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        self.assertEqual(elimination_order, ["sat", "diff", "grade"])


class TestMinWeight(BaseEliminationTest):
    def setUp(self):
        super(TestMinWeight, self).setUp()
        self.elimination_order = MinWeight(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost("diff"), 4)
        self.assertEqual(self.elimination_order.cost("intel"), 8)
        self.assertEqual(self.elimination_order.cost("reco"), 2)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        self.assertTrue(elimination_order[0] in ["sat", "reco"])
        self.assertTrue(elimination_order[1] in ["sat", "reco"])
        self.assertEqual(set(elimination_order[2:]), {"diff", "intel", "grade"})

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        self.assertEqual(elimination_order, ["sat", "diff", "grade"])


class TestMinFill(BaseEliminationTest):
    def setUp(self):
        super(TestMinFill, self).setUp()
        self.elimination_order = MinFill(self.model)

    def test_cost(self):
        self.assertEqual(self.elimination_order.cost("diff"), 0)
        self.assertEqual(self.elimination_order.cost("intel"), 1)
        self.assertEqual(self.elimination_order.cost("sat"), 0)

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        self.assertEqual(
            set(elimination_order), {"diff", "grade", "sat", "reco", "intel"}
        )

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "intel"], show_progress=False
        )
        self.assertEqual(set(elimination_order), {"diff", "grade", "intel"})
