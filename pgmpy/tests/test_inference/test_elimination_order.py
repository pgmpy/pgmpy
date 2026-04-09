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
    model = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade"), ("intel", "sat"), ("grade", "reco")])
    raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
    data = pd.DataFrame(raw_data, columns=["diff", "grade", "intel", "sat", "reco"])
    model.fit(data)
    return model


@pytest.fixture
def base_elimination(model):
    return BaseEliminationOrder(model)


@pytest.fixture
def weighted_min_fill(model):
    return WeightedMinFill(model)


@pytest.fixture
def min_neighbors(model):
    return MinNeighbors(model)


@pytest.fixture
def min_weight(model):
    return MinWeight(model)


@pytest.fixture
def min_fill(model):
    return MinFill(model)


class TestBaseElimination:
    def test_cost(self, base_elimination):
        costs = {"diff": 0, "sat": 0, "reco": 0, "grade": 0, "intel": 0}
        for var, expected_cost in costs.items():
            assert base_elimination.cost(var) == expected_cost

    def test_fill_in_edges(self, base_elimination):
        assert list(base_elimination.fill_in_edges("diff")) == []


class TestWeightedMinFill:
    def test_cost(self, weighted_min_fill):
        costs = {"diff": 4, "sat": 0, "reco": 0, "grade": 12, "intel": 12}
        for var, expected_cost in costs.items():
            assert weighted_min_fill.cost(var) == expected_cost

    def test_elimination_order(self, weighted_min_fill):
        elimination_order = weighted_min_fill.get_elimination_order(show_progress=False)
        assert set(elimination_order[:2]) == {"sat", "reco"}
        assert set(elimination_order[2:]) == {"grade", "intel", "diff"}

    def test_elimination_order_given_nodes(self, weighted_min_fill):
        elimination_order = weighted_min_fill.get_elimination_order(nodes=["diff", "grade", "sat"], show_progress=False)
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinNeighbors:
    def test_cost(self, min_neighbors):
        assert min_neighbors.cost("grade") == 3
        assert min_neighbors.cost("reco") == 1
        assert min_neighbors.cost("intel") == 3

    def test_elimination_order(self, min_neighbors):
        elimination_order = min_neighbors.get_elimination_order(show_progress=False)
        assert set(elimination_order[:2]) == {"sat", "reco"}
        assert set(elimination_order[2:]) == {"diff", "grade", "intel"}

    def test_elimination_order_given_nodes(self, min_neighbors):
        elimination_order = min_neighbors.get_elimination_order(nodes=["diff", "grade", "sat"], show_progress=False)
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinWeight:
    def test_cost(self, min_weight):
        assert min_weight.cost("diff") == 4
        assert min_weight.cost("intel") == 8
        assert min_weight.cost("reco") == 2

    def test_elimination_order(self, min_weight):
        elimination_order = min_weight.get_elimination_order(show_progress=False)
        assert elimination_order[0] in ["sat", "reco"]
        assert elimination_order[1] in ["sat", "reco"]
        assert set(elimination_order[2:]) == {"diff", "intel", "grade"}

    def test_elimination_order_given_nodes(self, min_weight):
        elimination_order = min_weight.get_elimination_order(nodes=["diff", "grade", "sat"], show_progress=False)
        assert elimination_order == ["sat", "diff", "grade"]


class TestMinFill:
    def test_cost(self, min_fill):
        assert min_fill.cost("diff") == 0
        assert min_fill.cost("intel") == 1
        assert min_fill.cost("sat") == 0

    def test_elimination_order(self, min_fill):
        elimination_order = min_fill.get_elimination_order(show_progress=False)
        assert set(elimination_order) == {"diff", "grade", "sat", "reco", "intel"}

    def test_elimination_order_given_nodes(self, min_fill):
        elimination_order = min_fill.get_elimination_order(nodes=["diff", "grade", "intel"], show_progress=False)
        assert set(elimination_order) == {"diff", "grade", "intel"}
