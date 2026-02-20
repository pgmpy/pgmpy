import numpy as np
import pandas as pd
import pytest

from pgmpy.inference.EliminationOrder import (
    BaseEliminationOrder,
    MinFill,
    MinNeighbors,
    MinWeight,
    WeightedMinFill,
)
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def model():
    model = DiscreteBayesianNetwork(
        [("diff", "grade"), ("intel", "grade"), ("intel", "sat"), ("grade", "reco")]
    )
    raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
    data = pd.DataFrame(raw_data, columns=["diff", "grade", "intel", "sat", "reco"])
    model.fit(data)
    return model


class TestBaseElimination:
    @pytest.fixture(autouse=True)
    def setUp(self, model):
        self.model = model
        self.elimination_order = BaseEliminationOrder(self.model)

    def test_cost(self):
        costs = {"diff": 0, "sat": 0, "reco": 0, "grade": 0, "intel": 0}
        for var, expected_cost in costs.items():
            assert self.elimination_order.cost(var) == expected_cost

    def test_fill_in_edges(self):
        assert list(self.elimination_order.fill_in_edges("diff")) == []


class TestWeightedMinFill:
    @pytest.fixture(autouse=True)
    def setUp(self, model):
        self.model = model
        self.elimination_order = WeightedMinFill(self.model)

    def test_cost(self):
        costs = {"diff": 4, "sat": 0, "reco": 0, "grade": 12, "intel": 12}
        for var, expected_cost in costs.items():
            assert self.elimination_order.cost(var) == expected_cost

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        assert set(elimination_order[:2]) == {"sat", "reco"}
        assert set(elimination_order[2:]) == {"grade", "intel", "diff"}

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinNeighbors:
    @pytest.fixture(autouse=True)
    def setUp(self, model):
        self.model = model
        self.elimination_order = MinNeighbors(self.model)

    def test_cost(self):
        assert self.elimination_order.cost("grade") == 3
        assert self.elimination_order.cost("reco") == 1
        assert self.elimination_order.cost("intel") == 3

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        assert set(elimination_order[:2]) == {"sat", "reco"}
        assert set(elimination_order[2:]) == {"diff", "grade", "intel"}

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinWeight:
    @pytest.fixture(autouse=True)
    def setUp(self, model):
        self.model = model
        self.elimination_order = MinWeight(self.model)

    def test_cost(self):
        assert self.elimination_order.cost("diff") == 4
        assert self.elimination_order.cost("intel") == 8
        assert self.elimination_order.cost("reco") == 2

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        assert elimination_order[0] in ["sat", "reco"]
        assert elimination_order[1] in ["sat", "reco"]
        assert set(elimination_order[2:]) == {"diff", "intel", "grade"}

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinFill:
    @pytest.fixture(autouse=True)
    def setUp(self, model):
        self.model = model
        self.elimination_order = MinFill(self.model)

    def test_cost(self):
        assert self.elimination_order.cost("diff") == 0
        assert self.elimination_order.cost("intel") == 1
        assert self.elimination_order.cost("sat") == 0

    def test_elimination_order(self):
        elimination_order = self.elimination_order.get_elimination_order(
            show_progress=False
        )
        assert set(elimination_order) == {"diff", "grade", "sat", "reco", "intel"}

    def test_elimination_order_given_nodes(self):
        elimination_order = self.elimination_order.get_elimination_order(
            nodes=["diff", "grade", "intel"], show_progress=False
        )
        assert set(elimination_order) == {"diff", "grade", "intel"}
