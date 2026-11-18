#!/usr/bin/env python3


import pytest

from pgmpy.base import UndirectedGraph
from pgmpy.tests import help_functions as hf


class TestUndirectedGraphCreation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = UndirectedGraph()

    def test_class_init_without_data(self):
        assert isinstance(self.graph, UndirectedGraph)

    def test_class_init_with_data_string(self):
        self.G = UndirectedGraph([("a", "b"), ("b", "c")])
        assert sorted(self.G.nodes()) == ["a", "b", "c"]
        assert hf.recursive_sorted(self.G.edges()) == [["a", "b"], ["b", "c"]]

    def test_add_node_string(self):
        self.graph.add_node("a")
        assert list(self.graph.nodes()) == ["a"]

    def test_add_node_nonstring(self):
        self.graph.add_node(1)
        assert list(self.graph.nodes()) == [1]

    def test_add_nodes_from_string(self):
        self.graph.add_nodes_from(["a", "b", "c", "d"])
        assert sorted(self.graph.nodes()) == ["a", "b", "c", "d"]

    def test_add_node_with_weight(self):
        self.graph.add_node("a")
        self.graph.add_node("weight_a", weight=0.3)
        assert self.graph.nodes["weight_a"]["weight"] == 0.3

    def test_add_nodes_from_non_string(self):
        self.graph.add_nodes_from([1, 2, 3, 4])

    def test_add_edge_string(self):
        self.graph.add_edge("d", "e")
        assert sorted(self.graph.nodes()) == ["d", "e"]
        assert hf.recursive_sorted(self.graph.edges()) == [["d", "e"]]
        self.graph.add_nodes_from(["a", "b", "c"])
        self.graph.add_edge("a", "b")
        assert hf.recursive_sorted(self.graph.edges()) == [["a", "b"], ["d", "e"]]

    def test_add_edge_nonstring(self):
        self.graph.add_edge(1, 2)

    def test_add_edges_from_string(self):
        self.graph.add_edges_from([("a", "b"), ("b", "c")])
        assert sorted(self.graph.nodes()) == ["a", "b", "c"]
        assert hf.recursive_sorted(self.graph.edges()) == [["a", "b"], ["b", "c"]]
        self.graph.add_nodes_from(["d", "e", "f"])
        self.graph.add_edges_from([("d", "e"), ("e", "f")])
        assert sorted(self.graph.nodes()) == ["a", "b", "c", "d", "e", "f"]
        assert hf.recursive_sorted(self.graph.edges()) == hf.recursive_sorted(
            [("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")]
        )

    def test_add_edges_from_nonstring(self):
        self.graph.add_edges_from([(1, 2), (2, 3)])

    def test_number_of_neighbors(self):
        self.graph.add_edges_from([("a", "b"), ("b", "c")])
        assert len(list(self.graph.neighbors("b"))) == 2


class TestUndirectedGraphMethods:
    def test_is_clique(self):
        G = UndirectedGraph(
            [
                ("A", "B"),
                ("C", "B"),
                ("B", "D"),
                ("B", "E"),
                ("D", "E"),
                ("E", "F"),
                ("D", "F"),
                ("B", "F"),
            ]
        )
        assert not G.is_clique(nodes=["A", "B", "C", "D"])
        assert G.is_clique(nodes=["B", "D", "E", "F"])
        assert G.is_clique(nodes=["D", "E", "B"])

    def test_is_triangulated(self):
        G = UndirectedGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        assert not G.is_triangulated()
        G.add_edge("A", "D")
        assert G.is_triangulated()
