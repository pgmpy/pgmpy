from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.independencies import Independencies
from pgmpy.models import NaiveBayes


class TestBaseModelCreation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.G = NaiveBayes()

    def test_class_init_without_data(self):
        assert isinstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = NaiveBayes(feature_vars=["b", "c"], dependent_var="a")
        assert Counter(list(self.g.nodes())) == Counter(["a", "b", "c"])
        assert Counter(list(self.g.edges())) == Counter([("a", "b"), ("a", "c")])
        assert self.g.dependent == "a"
        assert self.g.features == {"b", "c"}

    def test_class_init_with_data_nonstring(self):
        self.g = NaiveBayes(feature_vars=[2, 3], dependent_var=1)
        assert Counter(list(self.g.nodes())) == Counter([1, 2, 3])
        assert Counter(list(self.g.edges())) == Counter([(1, 2), (1, 3)])
        assert self.g.dependent == 1
        assert self.g.features == {2, 3}

    def test_add_node_string(self):
        self.G.add_node("a")
        assert list(self.G.nodes()) == ["a"]

    def test_add_node_nonstring(self):
        self.G.add_node(1)
        assert list(self.G.nodes()) == [1]

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(["a", "b", "c", "d"])
        assert Counter(list(self.G.nodes())) == Counter(["a", "b", "c", "d"])

    def test_add_nodes_from_non_string(self):
        self.G.add_nodes_from([1, 2, 3, 4])
        assert Counter(list(self.G.nodes())) == Counter([1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge("a", "b")
        assert Counter(list(self.G.nodes())) == Counter(["a", "b"])
        assert list(self.G.edges()) == [("a", "b")]
        assert self.G.dependent == "a"
        assert self.G.features == {"b"}

        self.G.add_nodes_from(["c", "d"])
        self.G.add_edge("a", "c")
        self.G.add_edge("a", "d")
        assert Counter(list(self.G.nodes())) == Counter(["a", "b", "c", "d"])
        assert Counter(list(self.G.edges())) == Counter([("a", "b"), ("a", "c"), ("a", "d")])
        assert self.G.dependent == "a"
        assert self.G.features == {"b", "c", "d"}

        with pytest.raises(ValueError):
            self.G.add_edge("b", "c")
        with pytest.raises(ValueError):
            self.G.add_edge("d", "f")
        with pytest.raises(ValueError):
            self.G.add_edge("e", "f")
        with pytest.raises(ValueError):
            self.G.add_edges_from([("a", "e"), ("b", "f")])
        with pytest.raises(ValueError):
            self.G.add_edges_from([("b", "f")])

    def test_add_edge_nonstring(self):
        self.G.add_edge(1, 2)
        assert Counter(list(self.G.nodes())) == Counter([1, 2])
        assert list(self.G.edges()) == [(1, 2)]
        assert self.G.dependent == 1
        assert self.G.features == {2}

        self.G.add_nodes_from([3, 4])
        self.G.add_edge(1, 3)
        self.G.add_edge(1, 4)
        assert Counter(list(self.G.nodes())) == Counter([1, 2, 3, 4])
        assert Counter(list(self.G.edges())) == Counter([(1, 2), (1, 3), (1, 4)])
        assert self.G.dependent == 1
        assert self.G.features == {2, 3, 4}

        with pytest.raises(ValueError):
            self.G.add_edge(2, 3)
        with pytest.raises(ValueError):
            self.G.add_edge(3, 6)
        with pytest.raises(ValueError):
            self.G.add_edge(5, 6)
        with pytest.raises(ValueError):
            self.G.add_edges_from([(1, 5), (2, 6)])
        with pytest.raises(ValueError):
            self.G.add_edges_from([(2, 6)])

    def test_add_edge_selfloop(self):
        with pytest.raises(ValueError):
            self.G.add_edge("a", "a")
        with pytest.raises(ValueError):
            self.G.add_edge(1, 1)

    def test_add_edges_from_self_loop(self):
        with pytest.raises(ValueError):
            self.G.add_edges_from([("a", "a")])

    def test_update_node_parents_bm_constructor(self):
        self.g = NaiveBayes(feature_vars=["b", "c"], dependent_var="a")
        assert list(self.g.predecessors("a")) == []
        assert list(self.g.predecessors("b")) == ["a"]
        assert list(self.g.predecessors("c")) == ["a"]

    def test_update_node_parents(self):
        self.G.add_nodes_from(["a", "b", "c"])
        self.G.add_edges_from([("a", "b"), ("a", "c")])
        assert list(self.G.predecessors("a")) == []
        assert list(self.G.predecessors("b")) == ["a"]
        assert list(self.G.predecessors("c")) == ["a"]


class TestNaiveBayesMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.G1 = NaiveBayes(feature_vars=["b", "c", "d", "e"], dependent_var="a")
        self.G2 = NaiveBayes(feature_vars=["g", "l", "s"], dependent_var="d")

    def test_local_independencies(self):
        assert self.G1.local_independencies("a") == Independencies()
        assert self.G1.local_independencies("b") == Independencies(["b", ["e", "c", "d"], "a"])
        assert self.G1.local_independencies("c") == Independencies(["c", ["e", "b", "d"], "a"])
        assert self.G1.local_independencies("d") == Independencies(["d", ["b", "c", "e"], "a"])

    def test_active_trail_nodes(self):
        assert sorted(self.G2.active_trail_nodes("d")) == ["d", "g", "l", "s"]
        assert sorted(self.G2.active_trail_nodes("g")) == ["d", "g", "l", "s"]
        assert sorted(self.G2.active_trail_nodes("l")) == ["d", "g", "l", "s"]
        assert sorted(self.G2.active_trail_nodes("s")) == ["d", "g", "l", "s"]

    def test_active_trail_nodes_args(self):
        assert sorted(self.G2.active_trail_nodes("d", observed="g")) == ["d", "l", "s"]
        assert sorted(self.G2.active_trail_nodes("l", observed="g")) == ["d", "l", "s"]
        assert sorted(self.G2.active_trail_nodes("s", observed=["g", "l"])) == ["d", "s"]
        assert sorted(self.G2.active_trail_nodes("s", observed=["d", "l"])) == ["s"]

    def test_get_ancestors(self):
        assert sorted(self.G1.get_ancestors("b")) == ["a", "b"]
        assert sorted(self.G1.get_ancestors("e")) == ["a", "e"]
        assert sorted(self.G1.get_ancestors("a")) == ["a"]
        assert sorted(self.G1.get_ancestors(["b", "e"])) == ["a", "b", "e"]


class TestNaiveBayesFit:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model1 = NaiveBayes()
        self.model2 = NaiveBayes(feature_vars=["B"], dependent_var="A")

    def test_fit_model_creation(self):
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        self.model1.fit(values, "A")
        assert Counter(list(self.model1.nodes())) == Counter(["A", "B", "C", "D", "E"])
        assert Counter(list(self.model1.edges())) == Counter([("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")])
        assert self.model1.dependent == "A"
        assert self.model1.features == {"B", "C", "D", "E"}

        self.model2.fit(values)
        assert Counter(list(self.model1.nodes())) == Counter(["A", "B", "C", "D", "E"])
        assert Counter(list(self.model1.edges())) == Counter([("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")])
        assert self.model2.dependent == "A"
        assert self.model2.features == {"B", "C", "D", "E"}

    def test_fit_model_creation_exception(self):
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        values2 = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 3)), columns=["C", "D", "E"])

        with pytest.raises(ValueError):
            self.model1.fit(values)
        with pytest.raises(ValueError):
            self.model1.fit(values2)
        with pytest.raises(ValueError):
            self.model2.fit(values2, "A")
