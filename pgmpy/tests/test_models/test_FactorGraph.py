from collections import Counter

import numpy as np
import pytest

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import DiscreteMarkovNetwork, FactorGraph, JunctionTree
from pgmpy.tests import help_functions as hf


class TestFactorGraphCreation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = FactorGraph()

    def test_class_init_without_data(self):
        assert isinstance(self.graph, FactorGraph)

    def test_class_init_data_string(self):
        self.graph = FactorGraph([("a", "phi1"), ("b", "phi1")])
        assert sorted(self.graph.nodes()) == ["a", "b", "phi1"]
        assert hf.recursive_sorted(self.graph.edges()) == [["a", "phi1"], ["b", "phi1"]]

    def test_add_single_node(self):
        self.graph.add_node("phi1")
        assert list(self.graph.nodes()) == ["phi1"]

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from(["a", "b", "phi1"])
        assert sorted(self.graph.nodes()) == ["a", "b", "phi1"]

    def test_add_single_edge(self):
        self.graph.add_edge("a", "phi1")
        assert sorted(self.graph.nodes()) == ["a", "phi1"]
        assert hf.recursive_sorted(self.graph.edges()) == [["a", "phi1"]]

    def test_add_multiple_edges(self):
        self.graph.add_edges_from([("a", "phi1"), ("b", "phi1")])
        assert sorted(self.graph.nodes()) == ["a", "b", "phi1"]
        assert hf.recursive_sorted(self.graph.edges()) == [["a", "phi1"], ["b", "phi1"]]

    def test_add_self_loop_raises_error(self):
        with pytest.raises(ValueError):
            self.graph.add_edge("a", "a")


class TestFactorGraphFactorOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = FactorGraph()

    def test_add_single_factor(self):
        self.graph.add_edges_from([("a", "phi1"), ("b", "phi1")])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        assert Counter(list(self.graph.factors)) == Counter([phi1])

    def test_add_multiple_factors(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        assert Counter(list(self.graph.factors)) == Counter([phi1, phi2])

    def test_get_factors(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])

        assert Counter(list(self.graph.get_factors())) == Counter([])

        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_factors(node=phi1) == phi1
        assert self.graph.get_factors(node=phi2) == phi2
        assert Counter(list(self.graph.get_factors())) == Counter([phi1, phi2])

        self.graph.remove_factors(phi1)
        with pytest.raises(ValueError):
            self.graph.get_factors(node=phi1)

    def test_remove_factors(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        assert set(self.graph.factors) == {phi2}
        assert (("c", phi2) in self.graph.edges()) or ((phi2, "c") in self.graph.edges())
        assert (("b", phi2) in self.graph.edges()) or ((phi2, "b") in self.graph.edges())
        assert set(self.graph.nodes()) == {"a", "b", "c", phi2}

    def test_get_partition_function(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], range(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], range(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_partition_function() == 22.0

    def test_get_point_mass_message(self):
        self.graph.add_node("a")
        phi = DiscreteFactor(["a"], [3], np.random.rand(3))
        self.graph.add_factors(phi)
        self.graph.add_edge("a", phi)
        message = self.graph.get_point_mass_message("a", 0)
        assert (message == np.array([1, 0, 0])).all()

    def test_get_uniform_message(self):
        self.graph.add_node("a")
        phi = DiscreteFactor(["a"], [4], np.random.rand(4))
        self.graph.add_factors(phi)
        self.graph.add_edge("a", phi)
        message = self.graph.get_uniform_message("a")
        assert (message == np.array([0.25, 0.25, 0.25, 0.25])).all()


class TestFactorGraphMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = FactorGraph()

    def test_get_cardinality(self):
        self.graph.add_edges_from(
            [
                ("a", "phi1"),
                ("b", "phi1"),
                ("c", "phi2"),
                ("d", "phi2"),
                ("a", "phi3"),
                ("d", "phi3"),
            ]
        )

        assert self.graph.get_cardinality() == {}

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        assert self.graph.get_cardinality() == {"a": 1, "b": 2}
        self.graph.remove_factors(phi1)
        assert self.graph.get_cardinality() == {}

        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["c", "d"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_cardinality() == {"d": 2, "a": 2, "b": 2, "c": 1}

        phi3 = DiscreteFactor(["d", "a"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi3)
        assert self.graph.get_cardinality() == {"d": 1, "c": 1, "b": 2, "a": 2}

        self.graph.remove_factors(phi1, phi2, phi3)
        assert self.graph.get_cardinality() == {}

    def test_get_cardinality_with_node(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_cardinality("a") == 2
        assert self.graph.get_cardinality("b") == 2
        assert self.graph.get_cardinality("c") == 2

    def test_get_factor_nodes(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))

        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        assert Counter(list(self.graph.get_factor_nodes())) == Counter([phi1, phi2])

    def test_get_variable_nodes(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        assert Counter(list(self.graph.get_variable_nodes())) == Counter(["a", "b", "c"])

    def test_get_variable_nodes_raises_error(self):
        self.graph.add_edges_from([("a", "phi1"), ("b", "phi1"), ("b", "phi2"), ("c", "phi2")])
        with pytest.raises(ValueError):
            self.graph.get_variable_nodes()

    def test_to_markov_model(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        mm = self.graph.to_markov_model()
        assert isinstance(mm, DiscreteMarkovNetwork)
        assert sorted(mm.nodes()) == ["a", "b", "c"]
        assert hf.recursive_sorted(mm.edges()) == [["a", "b"], ["b", "c"]]
        assert sorted(mm.get_factors(), key=lambda x: x.scope()) == [phi1, phi2]

    def test_to_junction_tree(self):
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])

        self.graph.add_factors(phi1, phi2)
        jt = self.graph.to_junction_tree()
        assert isinstance(jt, JunctionTree)
        assert hf.recursive_sorted(jt.nodes()) == [["a", "b"], ["b", "c"]]
        assert len(jt.edges()) == 1

    def test_check_model(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        assert self.graph.check_model()

        phi1 = DiscreteFactor(["a", "b"], [4, 2], np.random.rand(8))
        self.graph.add_factors(phi1, replace=True)
        assert self.graph.check_model()

    def test_check_model1(self):
        self.graph.add_nodes_from(["a", "b", "c", "d"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        with pytest.raises(ValueError):
            self.graph.check_model()

        self.graph.remove_node("d")
        assert self.graph.check_model()

    def test_check_model2(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)

        self.graph.add_edges_from([("a", "b")])
        with pytest.raises(ValueError):
            self.graph.check_model()

        self.graph.add_edges_from([(phi1, phi2)])
        with pytest.raises(ValueError):
            self.graph.check_model()

        self.graph.remove_edges_from([("a", "b"), (phi1, phi2)])
        assert self.graph.check_model()

    def test_check_model3(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        phi3 = DiscreteFactor(["a", "c"], [2, 2], np.random.rand(4))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2, phi3)
        with pytest.raises(ValueError):
            self.graph.check_model()
        self.graph.remove_factors(phi3)
        assert self.graph.check_model()

    def test_check_model4(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [3, 2], np.random.rand(6))
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        self.graph.add_factors(phi1, phi2)
        with pytest.raises(ValueError):
            self.graph.check_model()

        phi3 = DiscreteFactor(["c", "a"], [4, 4], np.random.rand(16))
        self.graph.add_factors(phi3, replace=True)
        with pytest.raises(ValueError):
            self.graph.check_model()

    def test_copy(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.add_nodes_from([phi1, phi2])
        self.graph.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        graph_copy = self.graph.copy()
        assert isinstance(graph_copy, FactorGraph)
        assert graph_copy.check_model()
        assert self.graph.get_factors() == graph_copy.get_factors()
        self.graph.remove_factors(phi1, phi2)
        assert phi1 not in self.graph.factors and phi2 not in self.graph.factors
        assert phi1 in graph_copy.factors and phi2 in graph_copy.factors
        self.graph.add_factors(phi1, phi2)
        self.graph.factors[0] = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        assert self.graph.get_factors()[0] != graph_copy.get_factors()[0]
        assert self.graph.factors != graph_copy.factors
