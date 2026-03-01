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


@pytest.fixture
def elimination_order(model):
    elimination_order = BaseEliminationOrder(model)
    return elimination_order


@pytest.fixture
def weighted_min_fill_elimination_order(model):
    elimination_order = WeightedMinFill(model)
    return elimination_order


@pytest.fixture
def min_neighbors_elimination_order(model):
    elimination_order = MinNeighbors(model)
    return elimination_order


@pytest.fixture
def min_weight_elimination_order(model):
    elimination_order = MinWeight(model)
    return elimination_order


@pytest.fixture
def min_fill_elimination_order(model):
    elimination_order = MinFill(model)
    return elimination_order


class TestBaseElimination:
    def test_cost(self, elimination_order):
        costs = {"diff": 0, "sat": 0, "reco": 0, "grade": 0, "intel": 0}
        for var, expected_cost in costs.items():
            assert elimination_order.cost(var) == expected_cost

    def test_fill_in_edges(self, elimination_order):
        assert list(elimination_order.fill_in_edges("diff")) == []


class TestWeightedMinFill:
    def test_cost(self, weighted_min_fill_elimination_order):
        costs = {"diff": 4, "sat": 0, "reco": 0, "grade": 12, "intel": 12}
        for var, expected_cost in costs.items():
            assert weighted_min_fill_elimination_order.cost(var) == expected_cost

    def test_elimination_order(self, weighted_min_fill_elimination_order):
        elimination_order = weighted_min_fill_elimination_order.get_elimination_order(
            show_progress=False
        )
        assert set(elimination_order[:2]) == {"sat", "reco"}
        assert set(elimination_order[2:]) == {"grade", "intel", "diff"}

    def test_elimination_order_given_nodes(self, weighted_min_fill_elimination_order):
        elimination_order = weighted_min_fill_elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinNeighbors:
    def test_cost(self, min_neighbors_elimination_order):
        assert min_neighbors_elimination_order.cost("grade") == 3
        assert min_neighbors_elimination_order.cost("reco") == 1
        assert min_neighbors_elimination_order.cost("intel") == 3

    def test_elimination_order(self, min_neighbors_elimination_order):
        elimination_order = min_neighbors_elimination_order.get_elimination_order(
            show_progress=False
        )
        assert set(elimination_order[:2]) == {"sat", "reco"}
        assert set(elimination_order[2:]) == {"diff", "grade", "intel"}

    def test_elimination_order_given_nodes(self, min_neighbors_elimination_order):
        elimination_order = min_neighbors_elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinWeight:
    def test_cost(self, min_weight_elimination_order):
        assert min_weight_elimination_order.cost("diff") == 4
        assert min_weight_elimination_order.cost("intel") == 8
        assert min_weight_elimination_order.cost("reco") == 2

    def test_elimination_order(self, min_weight_elimination_order):
        elimination_order = min_weight_elimination_order.get_elimination_order(
            show_progress=False
        )
        assert elimination_order[0] in ["sat", "reco"]
        assert elimination_order[1] in ["sat", "reco"]
        assert set(elimination_order[2:]) == {"diff", "intel", "grade"}

    def test_elimination_order_given_nodes(self, min_weight_elimination_order):
        elimination_order = min_weight_elimination_order.get_elimination_order(
            nodes=["diff", "grade", "sat"], show_progress=False
        )
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinFill:
    def test_cost(self, min_fill_elimination_order):
        assert min_fill_elimination_order.cost("diff") == 0
        assert min_fill_elimination_order.cost("intel") == 1
        assert min_fill_elimination_order.cost("sat") == 0

    def test_elimination_order(self, min_fill_elimination_order):
        elimination_order = min_fill_elimination_order.get_elimination_order(
            show_progress=False
        )
        assert set(elimination_order) == {"diff", "grade", "sat", "reco", "intel"}

    def test_elimination_order_given_nodes(self, min_fill_elimination_order):
        elimination_order = min_fill_elimination_order.get_elimination_order(
            nodes=["diff", "grade", "intel"], show_progress=False
        )
        assert set(elimination_order) == {"diff", "grade", "intel"}
