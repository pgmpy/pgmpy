from collections import Counter

import numpy as np
import pytest

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import ClusterGraph
from pgmpy.tests import help_functions as hf


class TestClusterGraphCreation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = ClusterGraph()

    def test_add_single_node(self):
        self.graph.add_node(("a", "b"))
        assert list(self.graph.nodes()) == [("a", "b")]

    def test_add_single_node_raises_error(self):
        with pytest.raises(TypeError):
            self.graph.add_node("a")

    def test_add_multiple_nodes(self):
        self.graph.add_nodes_from([("a", "b"), ("b", "c")])
        assert hf.recursive_sorted(self.graph.nodes()) == [["a", "b"], ["b", "c"]]

    def test_add_single_edge(self):
        self.graph.add_edge(("a", "b"), ("b", "c"))
        assert hf.recursive_sorted(self.graph.nodes()) == [["a", "b"], ["b", "c"]]
        assert sorted([node for edge in self.graph.edges() for node in edge]) == [("a", "b"), ("b", "c")]

    def test_add_single_edge_raises_error(self):
        with pytest.raises(ValueError):
            self.graph.add_edge(("a", "b"), ("c", "d"))


class TestClusterGraphFactorOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = ClusterGraph()

    def test_add_single_factor(self):
        self.graph.add_node(("a", "b"))
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        assert Counter(list(self.graph.factors)) == Counter([phi1])

    def test_add_single_factor_raises_error(self):
        self.graph.add_node(("a", "b"))
        phi1 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        with pytest.raises(ValueError):
            self.graph.add_factors(phi1)

    def test_add_multiple_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        assert Counter(list(self.graph.factors)) == Counter([phi1, phi2])

    def test_get_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        assert Counter(list(self.graph.get_factors())) == Counter([])
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_factors(node=("b", "a")) == phi1
        assert self.graph.get_factors(node=("b", "c")) == phi2
        assert Counter(list(self.graph.get_factors())) == Counter([phi1, phi2])

    def test_remove_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        self.graph.remove_factors(phi1)
        assert Counter(list(self.graph.factors)) == Counter([phi2])

    def test_get_partition_function(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], range(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], range(4))
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_partition_function() == 22.0


class TestClusterGraphMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = ClusterGraph()

    def test_get_cardinality(self):
        self.graph.add_edges_from([(("a", "b", "c"), ("a", "b")), (("a", "b", "c"), ("a", "c"))])

        assert self.graph.get_cardinality() == {}

        phi1 = DiscreteFactor(["a", "b", "c"], [1, 2, 2], np.random.rand(4))
        self.graph.add_factors(phi1)
        assert self.graph.get_cardinality() == {"a": 1, "b": 2, "c": 2}
        self.graph.remove_factors(phi1)
        assert self.graph.get_cardinality() == {}

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_cardinality() == {"a": 1, "b": 2, "c": 2}

        phi3 = DiscreteFactor(["a", "c"], [1, 1], np.random.rand(1))
        self.graph.add_factors(phi3)
        assert self.graph.get_cardinality() == {"c": 1, "b": 2, "a": 1}

        self.graph.remove_factors(phi1, phi2, phi3)
        assert self.graph.get_cardinality() == {}

    def test_get_cardinality_with_node(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        assert self.graph.get_cardinality("a") == 1
        assert self.graph.get_cardinality("b") == 2
        assert self.graph.get_cardinality("c") == 2

    def test_check_model(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1, phi2)
        assert self.graph.check_model()

        self.graph.remove_factors(phi2)
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        assert self.graph.check_model()

    def test_check_model1(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c")), (("a", "c"), ("a", "d"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi1)
        with pytest.raises(ValueError):
            self.graph.check_model()
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        self.graph.add_factors(phi2)
        with pytest.raises(ValueError):
            self.graph.check_model()

    def test_check_model2(self):
        self.graph.add_edges_from([(("a", "b"), ("a", "c")), (("a", "c"), ("a", "d"))])

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [3, 3], np.random.rand(9))
        phi3 = DiscreteFactor(["a", "d"], [4, 4], np.random.rand(16))
        self.graph.add_factors(phi1, phi2, phi3)
        with pytest.raises(ValueError):
            self.graph.check_model()
        self.graph.remove_factors(phi2)
        phi2 = DiscreteFactor(["a", "c"], [1, 3], np.random.rand(3))
        self.graph.add_factors(phi2)
        with pytest.raises(ValueError):
            self.graph.check_model()
        self.graph.remove_factors(phi3)

        phi3 = DiscreteFactor(["a", "d"], [1, 4], np.random.rand(4))
        self.graph.add_factors(phi3)
        assert self.graph.check_model()

    def test_copy_with_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        graph_copy = self.graph.copy()
        assert isinstance(graph_copy, ClusterGraph)
        assert hf.recursive_sorted(self.graph.nodes()) == hf.recursive_sorted(graph_copy.nodes())
        assert hf.recursive_sorted(self.graph.edges()) == hf.recursive_sorted(graph_copy.edges())
        assert graph_copy.check_model()
        assert self.graph.get_factors() == graph_copy.get_factors()
        self.graph.remove_factors(phi1, phi2)
        assert phi1 not in self.graph.factors and phi2 not in self.graph.factors
        assert phi1 in graph_copy.factors and phi2 in graph_copy.factors
        self.graph.add_factors(phi1, phi2)
        self.graph.factors[0] = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        assert self.graph.get_factors()[0] != graph_copy.get_factors()[0]
        assert self.graph.factors != graph_copy.factors

    def test_copy_without_factors(self):
        self.graph.add_nodes_from([("a", "b", "c"), ("a", "b"), ("a", "c")])
        self.graph.add_edges_from([(("a", "b", "c"), ("a", "b")), (("a", "b", "c"), ("a", "c"))])
        graph_copy = self.graph.copy()
        self.graph.remove_edge(("a", "b", "c"), ("a", "c"))
        assert not self.graph.has_edge(("a", "b", "c"), ("a", "c"))
        assert graph_copy.has_edge(("a", "b", "c"), ("a", "c"))
        self.graph.remove_node(("a", "c"))
        assert not self.graph.has_node(("a", "c"))
        assert graph_copy.has_node(("a", "c"))
        self.graph.add_node(("c", "d"))
        assert self.graph.has_node(("c", "d"))
        assert not graph_copy.has_node(("c", "d"))
