import pytest

import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import ClusterGraph
from pgmpy.tests import help_functions as hf


@pytest.fixture
def setup_cluster_graph():
    """Fixture to set up a basic ClusterGraph for testing."""
    graph = ClusterGraph()
    yield graph
    del graph


@pytest.fixture
def setup_cluster_graph_with_edges():
    """Fixture to set up a ClusterGraph with edges for testing."""
    graph = ClusterGraph()
    graph.add_edges_from([[("a", "b"), ("b", "c")]])
    yield graph
    del graph


class TestClusterGraphCreation:
    def test_add_single_node(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_node(("a", "b"))
        assert list(graph.nodes()) == [("a", "b")]

    def test_add_single_node_raises_error(self, setup_cluster_graph):
        graph = setup_cluster_graph
        with pytest.raises(TypeError):
            graph.add_node("a")

    def test_add_multiple_nodes(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_nodes_from([("a", "b"), ("b", "c")])
        assert hf.recursive_sorted(graph.nodes()) == [["a", "b"], ["b", "c"]]

    def test_add_single_edge(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_edge(("a", "b"), ("b", "c"))
        assert hf.recursive_sorted(graph.nodes()) == [["a", "b"], ["b", "c"]]
        assert sorted([node for edge in graph.edges() for node in edge]) == [
            ("a", "b"),
            ("b", "c"),
        ]

    def test_add_single_edge_raises_error(self, setup_cluster_graph):
        graph = setup_cluster_graph
        with pytest.raises(ValueError):
            graph.add_edge(("a", "b"), ("c", "d"))


class TestClusterGraphFactorOperations:
    def test_add_single_factor(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_node(("a", "b"))
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        graph.add_factors(phi1)
        assert list(graph.factors) == [phi1]

    def test_add_single_factor_raises_error(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_node(("a", "b"))
        phi1 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        with pytest.raises(ValueError):
            graph.add_factors(phi1)

    def test_add_multiple_factors(self, setup_cluster_graph_with_edges):
        graph = setup_cluster_graph_with_edges
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        graph.add_factors(phi1, phi2)
        assert list(graph.factors) == [phi1, phi2]

    def test_get_factors(self, setup_cluster_graph_with_edges):
        graph = setup_cluster_graph_with_edges
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        assert graph.get_factors() == []
        graph.add_factors(phi1, phi2)
        assert graph.get_factors(node=("b", "a")) == phi1
        assert graph.get_factors(node=("b", "c")) == phi2
        assert list(graph.get_factors()) == [phi1, phi2]

    def test_remove_factors(self, setup_cluster_graph_with_edges):
        graph = setup_cluster_graph_with_edges
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        graph.add_factors(phi1, phi2)
        graph.remove_factors(phi1)
        assert list(graph.factors) == [phi2]

    def test_get_partition_function(self, setup_cluster_graph_with_edges):
        graph = setup_cluster_graph_with_edges
        phi1 = DiscreteFactor(["a", "b"], [2, 2], range(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], range(4))
        graph.add_factors(phi1, phi2)
        assert graph.get_partition_function() == 22.0


class TestClusterGraphMethods:
    def test_get_cardinality(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_edges_from(
            [(("a", "b", "c"), ("a", "b")), (("a", "b", "c"), ("a", "c"))]
        )

        assert graph.get_cardinality() == {}

        phi1 = DiscreteFactor(["a", "b", "c"], [1, 2, 2], np.random.rand(4))
        graph.add_factors(phi1)
        assert graph.get_cardinality() == {"a": 1, "b": 2, "c": 2}
        graph.remove_factors(phi1)
        assert graph.get_cardinality() == {}

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        graph.add_factors(phi1, phi2)
        assert graph.get_cardinality() == {"a": 1, "b": 2, "c": 2}

        phi3 = DiscreteFactor(["a", "c"], [1, 1], np.random.rand(1))
        graph.add_factors(phi3)
        assert graph.get_cardinality() == {"c": 1, "b": 2, "a": 1}

        graph.remove_factors(phi1, phi2, phi3)
        assert graph.get_cardinality() == {}

    def test_get_cardinality_with_node(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_edges_from([(("a", "b"), ("a", "c"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        graph.add_factors(phi1, phi2)
        assert graph.get_cardinality("a") == 1
        assert graph.get_cardinality("b") == 2
        assert graph.get_cardinality("c") == 2

    def test_check_model(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_edges_from([(("a", "b"), ("a", "c"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        graph.add_factors(phi1, phi2)
        assert graph.check_model() is True

        graph.remove_factors(phi2)
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        graph.add_factors(phi2)
        assert graph.check_model() is True

    def test_check_model1(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_edges_from([(("a", "b"), ("a", "c")), (("a", "c"), ("a", "d"))])
        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        graph.add_factors(phi1)
        with pytest.raises(ValueError):
            graph.check_model()
        phi2 = DiscreteFactor(["a", "c"], [1, 2], np.random.rand(2))
        graph.add_factors(phi2)
        with pytest.raises(ValueError):
            graph.check_model()

    def test_check_model2(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_edges_from([(("a", "b"), ("a", "c")), (("a", "c"), ("a", "d"))])

        phi1 = DiscreteFactor(["a", "b"], [1, 2], np.random.rand(2))
        phi2 = DiscreteFactor(["a", "c"], [3, 3], np.random.rand(9))
        phi3 = DiscreteFactor(["a", "d"], [4, 4], np.random.rand(16))
        graph.add_factors(phi1, phi2, phi3)
        with pytest.raises(ValueError):
            graph.check_model()
        graph.remove_factors(phi2)
        phi2 = DiscreteFactor(["a", "c"], [1, 3], np.random.rand(3))
        graph.add_factors(phi2)
        with pytest.raises(ValueError):
            graph.check_model()
        graph.remove_factors(phi3)

        phi3 = DiscreteFactor(["a", "d"], [1, 4], np.random.rand(4))
        graph.add_factors(phi3)
        assert graph.check_model() is True

    def test_copy_with_factors(self, setup_cluster_graph_with_edges):
        graph = setup_cluster_graph_with_edges
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        graph.add_factors(phi1, phi2)
        graph_copy = graph.copy()
        assert isinstance(graph_copy, ClusterGraph)
        assert hf.recursive_sorted(graph.nodes()) == hf.recursive_sorted(
            graph_copy.nodes()
        )
        assert hf.recursive_sorted(graph.edges()) == hf.recursive_sorted(
            graph_copy.edges()
        )
        assert graph_copy.check_model() is True
        assert graph.get_factors() == graph_copy.get_factors()
        graph.remove_factors(phi1, phi2)
        assert phi1 not in graph.factors and phi2 not in graph.factors
        assert phi1 in graph_copy.factors and phi2 in graph_copy.factors
        graph.add_factors(phi1, phi2)
        graph.factors[0] = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        assert graph.get_factors()[0] != graph_copy.get_factors()[0]
        assert graph.factors != graph_copy.factors

    def test_copy_without_factors(self, setup_cluster_graph):
        graph = setup_cluster_graph
        graph.add_nodes_from([("a", "b", "c"), ("a", "b"), ("a", "c")])
        graph.add_edges_from(
            [(("a", "b", "c"), ("a", "b")), (("a", "b", "c"), ("a", "c"))]
        )
        graph_copy = graph.copy()
        graph.remove_edge(("a", "b", "c"), ("a", "c"))
        assert graph.has_edge(("a", "b", "c"), ("a", "c")) is False
        assert graph_copy.has_edge(("a", "b", "c"), ("a", "c")) is True
        graph.remove_node(("a", "c"))
        assert graph.has_node(("a", "c")) is False
        assert graph_copy.has_node(("a", "c")) is True
        graph.add_node(("c", "d"))
        assert graph.has_node(("c", "d")) is True
        assert graph_copy.has_node(("c", "d")) is False
