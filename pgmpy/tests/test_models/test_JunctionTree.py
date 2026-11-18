import numpy as np
import pytest

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import JunctionTree
from pgmpy.tests import help_functions as hf


class TestJunctionTreeCreation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = JunctionTree()

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

    def test_add_cyclic_path_raises_error(self):
        self.graph.add_edge(("a", "b"), ("b", "c"))
        self.graph.add_edge(("b", "c"), ("c", "d"))
        with pytest.raises(ValueError):
            self.graph.add_edge(("c", "d"), ("a", "b"))


class TestJunctionTreeMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.factor1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        self.factor2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.factor3 = DiscreteFactor(["d", "e"], [2, 2], np.random.rand(4))
        self.factor4 = DiscreteFactor(["e", "f"], [2, 2], np.random.rand(4))
        self.factor5 = DiscreteFactor(["a", "b", "e"], [2, 2, 2], np.random.rand(8))

        self.graph1 = JunctionTree()
        self.graph1.add_edge(("a", "b"), ("b", "c"))
        self.graph1.add_factors(self.factor1, self.factor2)

        self.graph2 = JunctionTree()
        self.graph2.add_nodes_from([("a", "b"), ("b", "c"), ("d", "e")])
        self.graph2.add_edge(("a", "b"), ("b", "c"))
        self.graph2.add_factors(self.factor1, self.factor2, self.factor3)

        self.graph3 = JunctionTree()
        self.graph3.add_edges_from([(("a", "b"), ("b", "c")), (("d", "e"), ("e", "f"))])
        self.graph3.add_factors(self.factor1, self.factor2, self.factor3, self.factor4)

        self.graph4 = JunctionTree()
        self.graph4.add_edges_from(
            [
                (("a", "b", "e"), ("b", "c")),
                (("a", "b", "e"), ("e", "f")),
                (("d", "e"), ("e", "f")),
            ]
        )
        self.graph4.add_factors(self.factor5, self.factor2, self.factor3, self.factor4)

    def test_check_model(self):
        with pytest.raises(ValueError):
            self.graph2.check_model()
        with pytest.raises(ValueError):
            self.graph3.check_model()
        assert self.graph1.check_model()
        assert self.graph4.check_model()

    def test_states(self):
        assert self.graph4.states == {
            "a": [0, 1],
            "b": [0, 1],
            "e": [0, 1],
            "c": [0, 1],
            "d": [0, 1],
            "f": [0, 1],
        }


class TestJunctionTreeCopy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = JunctionTree()

    def test_copy_with_nodes(self):
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

    def test_copy_with_factors(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        graph_copy = self.graph.copy()

        assert isinstance(graph_copy, JunctionTree)
        assert self.graph is not graph_copy
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

    def test_copy_with_factorchanges(self):
        self.graph.add_edges_from([[("a", "b"), ("b", "c")]])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        self.graph.add_factors(phi1, phi2)
        graph_copy = self.graph.copy()

        self.graph.factors[0].reduce([("a", 0)])
        assert self.graph.factors[0].scope() != graph_copy.factors[0].scope()
        assert self.graph != graph_copy
        self.graph.factors[1].marginalize(["b"])
        assert self.graph.factors[1].scope() != graph_copy.factors[1].scope()
        assert self.graph != graph_copy
