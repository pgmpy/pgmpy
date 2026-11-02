import numpy as np
import pandas as pd
import pytest

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.EliminationOrder import (
    ELIMINATION_HEURISTICS,
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
    BaseEliminationOrder,
    MinFill,
    MinNeighbors,
    MinWeight,
    WeightedMinFill,
)
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork


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
        costs = {"diff": 0, "sat": 0, "reco": 0, "grade": 8, "intel": 8}
        for var, expected_cost in costs.items():
            assert weighted_min_fill.cost(var) == expected_cost

    def test_elimination_order(self, weighted_min_fill):
        elimination_order = weighted_min_fill.get_elimination_order(show_progress=False)
        assert elimination_order == ["diff", "sat", "intel", "grade", "reco"]

    def test_elimination_order_given_nodes(self, weighted_min_fill):
        elimination_order = weighted_min_fill.get_elimination_order(nodes=["diff", "grade", "sat"], show_progress=False)
        assert elimination_order == ["diff", "sat", "grade"]


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
        assert min_fill.cost("intel") == 2
        assert min_fill.cost("sat") == 0

    def test_fill_in_edges_are_added_during_ordering(self, min_fill):
        assert set(map(frozenset, min_fill.fill_in_edges("grade"))) == {
            frozenset({"diff", "reco"}),
            frozenset({"intel", "reco"}),
        }
        assert min_fill.get_elimination_order(nodes=["grade"], show_progress=False) == ["grade"]
        assert min_fill.moralized_model.has_edge("diff", "reco")
        assert min_fill.moralized_model.has_edge("intel", "reco")
        assert "grade" not in min_fill.moralized_model.nodes()

    def test_elimination_order(self, min_fill):
        elimination_order = min_fill.get_elimination_order(show_progress=False)
        assert set(elimination_order) == {"diff", "grade", "sat", "reco", "intel"}

    def test_elimination_order_given_nodes(self, min_fill):
        elimination_order = min_fill.get_elimination_order(nodes=["diff", "grade", "intel"], show_progress=False)
        assert set(elimination_order) == {"diff", "grade", "intel"}


class TestEliminationOrderOnMarkovNetwork:
    def setup_method(self):
        self.graph = DiscreteMarkovNetwork([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
        self.graph.add_factors(
            DiscreteFactor(["a", "b"], [2, 3], np.ones(6)),
            DiscreteFactor(["b", "c"], [3, 4], np.ones(12)),
            DiscreteFactor(["c", "d"], [4, 5], np.ones(20)),
            DiscreteFactor(["d", "a"], [5, 2], np.ones(10)),
        )

    def test_fill_in_edges(self):
        min_fill = MinFill(self.graph)
        assert list(min_fill.fill_in_edges("a")) == [("b", "d")]
        min_fill.moralized_model.add_edge("b", "d")
        assert list(min_fill.fill_in_edges("a")) == []
        assert not self.graph.has_edge("b", "d")

    def test_heuristic_costs(self):
        expected = {
            "minfill": {"a": 1, "b": 1, "c": 1, "d": 1},
            "weightedminfill": {"a": 15, "b": 8, "c": 15, "d": 8},
            "minneighbors": {"a": 2, "b": 2, "c": 2, "d": 2},
            "minweight": {"a": 15, "b": 8, "c": 15, "d": 8},
            "h1": {"a": 15, "b": 8, "c": 15, "d": 8},
            "h2": {"a": 7.5, "b": 8 / 3, "c": 3.75, "d": 1.6},
            "h3": {"a": 15 - 10, "b": 8 - 12, "c": 15 - 20, "d": 8 - 20},
            "h4": {"a": 15 - 16, "b": 8 - 18, "c": 15 - 32, "d": 8 - 30},
            "h5": {"a": 15 / 10, "b": 8 / 12, "c": 15 / 20, "d": 8 / 20},
            "h6": {"a": 15 / 16, "b": 8 / 18, "c": 15 / 32, "d": 8 / 30},
        }
        assert set(ELIMINATION_HEURISTICS) == set(expected)
        assert ELIMINATION_HEURISTICS["h6"] is H6 and ELIMINATION_HEURISTICS["minfill"] is MinFill
        for heuristic, costs in expected.items():
            elimination_order = ELIMINATION_HEURISTICS[heuristic](self.graph)
            for node, cost in costs.items():
                assert elimination_order.cost(node) == pytest.approx(cost), (heuristic, node)

    def test_missing_cardinality_counts_as_one(self):
        graph = DiscreteMarkovNetwork([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
        assert MinWeight(graph).cost("a") == 1
        assert H6(graph).cost("a") == 0.5

    def test_order_is_greedy_and_deterministic(self):
        assert MinFill(self.graph).get_elimination_order(show_progress=False) == ["a", "b", "c", "d"]
        assert H1(self.graph).get_elimination_order(show_progress=False) == ["b", "d", "c", "a"]
        assert MinWeight(self.graph).get_elimination_order(show_progress=False) == ["b", "d", "c", "a"]
        for heuristic in [H2, H3, H4, H5, H6]:
            assert heuristic(self.graph).get_elimination_order(show_progress=False)[0] == "d"

    def test_nodes_argument_restricts_elimination(self):
        min_fill = MinFill(self.graph)
        assert min_fill.get_elimination_order(nodes=["c", "a"], show_progress=False) == ["a", "c"]
        assert set(min_fill.moralized_model.nodes()) == {"b", "d"}
        assert min_fill.moralized_model.has_edge("b", "d")
        with pytest.raises(ValueError):
            MinFill(self.graph).get_elimination_order(nodes=["zzz"], show_progress=False)

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError):
            MinFill("not a model")
