import pytest
import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import JunctionTree
from pgmpy.tests import help_functions as hf


@pytest.fixture
def setup_junction_tree_creation():
    """Fixture to set up a basic JunctionTree model for testing creation."""
    graph = JunctionTree()
    yield graph
    # Cleanup
    del graph


@pytest.fixture
def setup_junction_tree_methods():
    """Fixture to set up JunctionTree models for testing methods."""
    factor1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
    factor2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
    factor3 = DiscreteFactor(["d", "e"], [2, 2], np.random.rand(4))
    factor4 = DiscreteFactor(["e", "f"], [2, 2], np.random.rand(4))
    factor5 = DiscreteFactor(["a", "b", "e"], [2, 2, 2], np.random.rand(8))

    graph1 = JunctionTree()
    graph1.add_edge(("a", "b"), ("b", "c"))
    graph1.add_factors(factor1, factor2)

    graph2 = JunctionTree()
    graph2.add_nodes_from([("a", "b"), ("b", "c"), ("d", "e")])
    graph2.add_edge(("a", "b"), ("b", "c"))
    graph2.add_factors(factor1, factor2, factor3)

    graph3 = JunctionTree()
    graph3.add_edges_from([(("a", "b"), ("b", "c")), (("d", "e"), ("e", "f"))])
    graph3.add_factors(factor1, factor2, factor3, factor4)

    graph4 = JunctionTree()
    graph4.add_edges_from(
        [
            (("a", "b", "e"), ("b", "c")),
            (("a", "b", "e"), ("e", "f")),
            (("d", "e"), ("e", "f")),
        ]
    )
    graph4.add_factors(factor5, factor2, factor3, factor4)

    yield {
        "factor1": factor1,
        "factor2": factor2,
        "factor3": factor3,
        "factor4": factor4,
        "factor5": factor5,
        "graph1": graph1,
        "graph2": graph2,
        "graph3": graph3,
        "graph4": graph4,
    }
    # Cleanup
    del factor1, factor2, factor3, factor4, factor5
    del graph1, graph2, graph3, graph4


@pytest.fixture
def setup_junction_tree_copy():
    """Fixture to set up a JunctionTree model for testing copy."""
    graph = JunctionTree()
    yield graph
    # Cleanup
    del graph


class TestJunctionTreeCreation:
    def test_add_single_node(self, setup_junction_tree_creation):
        graph = setup_junction_tree_creation
        graph.add_node(("a", "b"))
        assert list(graph.nodes()) == [("a", "b")]

    def test_add_single_node_raises_error(self, setup_junction_tree_creation):
        graph = setup_junction_tree_creation
        with pytest.raises(TypeError):
            graph.add_node("a")

    def test_add_multiple_nodes(self, setup_junction_tree_creation):
        graph = setup_junction_tree_creation
        graph.add_nodes_from([("a", "b"), ("b", "c")])
        assert hf.recursive_sorted(graph.nodes()) == [["a", "b"], ["b", "c"]]

    def test_add_single_edge(self, setup_junction_tree_creation):
        graph = setup_junction_tree_creation
        graph.add_edge(("a", "b"), ("b", "c"))
        assert hf.recursive_sorted(graph.nodes()) == [["a", "b"], ["b", "c"]]
        assert sorted([node for edge in graph.edges() for node in edge]) == [
            ("a", "b"),
            ("b", "c"),
        ]

    def test_add_single_edge_raises_error(self, setup_junction_tree_creation):
        graph = setup_junction_tree_creation
        with pytest.raises(ValueError):
            graph.add_edge(("a", "b"), ("c", "d"))

    def test_add_cyclic_path_raises_error(self, setup_junction_tree_creation):
        graph = setup_junction_tree_creation
        graph.add_edge(("a", "b"), ("b", "c"))
        graph.add_edge(("b", "c"), ("c", "d"))
        with pytest.raises(ValueError):
            graph.add_edge(("c", "d"), ("a", "b"))


class TestJunctionTreeMethods:
    def test_check_model(self, setup_junction_tree_methods):
        data = setup_junction_tree_methods
        with pytest.raises(ValueError):
            data["graph2"].check_model()
        with pytest.raises(ValueError):
            data["graph3"].check_model()
        assert data["graph1"].check_model() is True
        assert data["graph4"].check_model() is True

    def test_states(self, setup_junction_tree_methods):
        data = setup_junction_tree_methods
        assert data["graph4"].states == {
            "a": [0, 1],
            "b": [0, 1],
            "e": [0, 1],
            "c": [0, 1],
            "d": [0, 1],
            "f": [0, 1],
        }


class TestJunctionTreeCopy:
    def test_copy_with_nodes(self, setup_junction_tree_copy):
        graph = setup_junction_tree_copy
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

    def test_copy_with_factors(self, setup_junction_tree_copy):
        graph = setup_junction_tree_copy
        graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        graph.add_factors(phi1, phi2)
        graph_copy = graph.copy()

        assert isinstance(graph_copy, JunctionTree)
        assert graph is not graph_copy
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

    def test_copy_with_factorchanges(self, setup_junction_tree_copy):
        graph = setup_junction_tree_copy
        graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        graph.add_factors(phi1, phi2)
        graph_copy = graph.copy()

        graph.factors[0].reduce([("a", 0)])
        assert graph.factors[0].scope() != graph_copy.factors[0].scope()
        assert graph != graph_copy
        graph.factors[1].marginalize(["b"])
        assert graph.factors[1].scope() != graph_copy.factors[1].scope()
        assert graph != graph_copy
