import pytest

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.independencies import Independencies
from pgmpy.models import NaiveBayes


@pytest.fixture
def setup_base_model():
    """Fixture to set up a basic NaiveBayes model for testing."""
    G = NaiveBayes()
    yield G
    # Cleanup
    del G


@pytest.fixture
def setup_naive_bayes_models():
    """Fixture to set up NaiveBayes models for method testing."""
    G1 = NaiveBayes(feature_vars=["b", "c", "d", "e"], dependent_var="a")
    G2 = NaiveBayes(feature_vars=["g", "l", "s"], dependent_var="d")
    yield {"G1": G1, "G2": G2}
    # Cleanup
    del G1
    del G2


@pytest.fixture
def setup_fit_models():
    """Fixture to set up models for fit testing."""
    model1 = NaiveBayes()
    model2 = NaiveBayes(feature_vars=["B"], dependent_var="A")
    yield {"model1": model1, "model2": model2}
    # Cleanup
    del model1
    del model2


class TestBaseModelCreation:
    def test_class_init_without_data(self, setup_base_model):
        G = setup_base_model
        assert isinstance(G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        g = NaiveBayes(feature_vars=["b", "c"], dependent_var="a")
        assert sorted(list(g.nodes())) == sorted(["a", "b", "c"])
        assert sorted(list(g.edges())) == sorted([("a", "b"), ("a", "c")])
        assert g.dependent == "a"
        assert g.features == {"b", "c"}

    def test_class_init_with_data_nonstring(self):
        g = NaiveBayes(feature_vars=[2, 3], dependent_var=1)
        assert sorted(list(g.nodes())) == sorted([1, 2, 3])
        assert sorted(list(g.edges())) == sorted([(1, 2), (1, 3)])
        assert g.dependent == 1
        assert g.features == {2, 3}

    def test_add_node_string(self, setup_base_model):
        G = setup_base_model
        G.add_node("a")
        assert list(G.nodes()) == ["a"]

    def test_add_node_nonstring(self, setup_base_model):
        G = setup_base_model
        G.add_node(1)
        assert list(G.nodes()) == [1]

    def test_add_nodes_from_string(self, setup_base_model):
        G = setup_base_model
        G.add_nodes_from(["a", "b", "c", "d"])
        assert sorted(list(G.nodes())) == sorted(["a", "b", "c", "d"])

    def test_add_nodes_from_non_string(self, setup_base_model):
        G = setup_base_model
        G.add_nodes_from([1, 2, 3, 4])
        assert sorted(list(G.nodes())) == sorted([1, 2, 3, 4])

    def test_add_edge_string(self, setup_base_model):
        G = setup_base_model
        G.add_edge("a", "b")
        assert sorted(list(G.nodes())) == sorted(["a", "b"])
        assert list(G.edges()) == [("a", "b")]
        assert G.dependent == "a"
        assert G.features == {"b"}

        G.add_nodes_from(["c", "d"])
        G.add_edge("a", "c")
        G.add_edge("a", "d")
        assert sorted(list(G.nodes())) == sorted(["a", "b", "c", "d"])
        assert sorted(list(G.edges())) == sorted([("a", "b"), ("a", "c"), ("a", "d")])
        assert G.dependent == "a"
        assert G.features == {"b", "c", "d"}

        with pytest.raises(ValueError):
            G.add_edge("b", "c")
        with pytest.raises(ValueError):
            G.add_edge("d", "f")
        with pytest.raises(ValueError):
            G.add_edge("e", "f")
        with pytest.raises(ValueError):
            G.add_edges_from([("a", "e"), ("b", "f")])
        with pytest.raises(ValueError):
            G.add_edges_from([("b", "f")])

    def test_add_edge_nonstring(self, setup_base_model):
        G = setup_base_model
        G.add_edge(1, 2)
        assert sorted(list(G.nodes())) == sorted([1, 2])
        assert list(G.edges()) == [(1, 2)]
        assert G.dependent == 1
        assert G.features == {2}

        G.add_nodes_from([3, 4])
        G.add_edge(1, 3)
        G.add_edge(1, 4)
        assert sorted(list(G.nodes())) == sorted([1, 2, 3, 4])
        assert sorted(list(G.edges())) == sorted([(1, 2), (1, 3), (1, 4)])
        assert G.dependent == 1
        assert G.features == {2, 3, 4}

        with pytest.raises(ValueError):
            G.add_edge(2, 3)
        with pytest.raises(ValueError):
            G.add_edge(3, 6)
        with pytest.raises(ValueError):
            G.add_edge(5, 6)
        with pytest.raises(ValueError):
            G.add_edges_from([(1, 5), (2, 6)])
        with pytest.raises(ValueError):
            G.add_edges_from([(2, 6)])

    def test_add_edge_selfloop(self, setup_base_model):
        G = setup_base_model
        with pytest.raises(ValueError):
            G.add_edge("a", "a")
        with pytest.raises(ValueError):
            G.add_edge(1, 1)

    def test_add_edges_from_self_loop(self, setup_base_model):
        G = setup_base_model
        with pytest.raises(ValueError):
            G.add_edges_from([("a", "a")])

    def test_update_node_parents_bm_constructor(self):
        g = NaiveBayes(feature_vars=["b", "c"], dependent_var="a")
        assert list(g.predecessors("a")) == []
        assert list(g.predecessors("b")) == ["a"]
        assert list(g.predecessors("c")) == ["a"]

    def test_update_node_parents(self, setup_base_model):
        G = setup_base_model
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([("a", "b"), ("a", "c")])
        assert list(G.predecessors("a")) == []
        assert list(G.predecessors("b")) == ["a"]
        assert list(G.predecessors("c")) == ["a"]


class TestNaiveBayesMethods:
    def test_local_independencies(self, setup_naive_bayes_models):
        G1 = setup_naive_bayes_models["G1"]
        assert G1.local_independencies("a") == Independencies()
        assert G1.local_independencies("b") == Independencies(["b", ["e", "c", "d"], "a"])
        assert G1.local_independencies("c") == Independencies(["c", ["e", "b", "d"], "a"])
        assert G1.local_independencies("d") == Independencies(["d", ["b", "c", "e"], "a"])

    def test_active_trail_nodes(self, setup_naive_bayes_models):
        G2 = setup_naive_bayes_models["G2"]
        assert sorted(G2.active_trail_nodes("d")) == sorted(["d", "g", "l", "s"])
        assert sorted(G2.active_trail_nodes("g")) == sorted(["d", "g", "l", "s"])
        assert sorted(G2.active_trail_nodes("l")) == sorted(["d", "g", "l", "s"])
        assert sorted(G2.active_trail_nodes("s")) == sorted(["d", "g", "l", "s"])

    def test_active_trail_nodes_args(self, setup_naive_bayes_models):
        G2 = setup_naive_bayes_models["G2"]
        assert sorted(G2.active_trail_nodes("d", observed="g")) == sorted(["d", "l", "s"])
        assert sorted(G2.active_trail_nodes("l", observed="g")) == sorted(["d", "l", "s"])
        assert sorted(G2.active_trail_nodes("s", observed=["g", "l"])) == sorted(["d", "s"])
        assert sorted(G2.active_trail_nodes("s", observed=["d", "l"])) == sorted(["s"])

    def test_get_ancestors(self, setup_naive_bayes_models):
        G1 = setup_naive_bayes_models["G1"]
        assert sorted(G1.get_ancestors("b")) == sorted(["a", "b"])
        assert sorted(G1.get_ancestors("e")) == sorted(["a", "e"])
        assert sorted(G1.get_ancestors("a")) == sorted(["a"])
        assert sorted(G1.get_ancestors(["b", "e"])) == sorted(["a", "b", "e"])


class TestNaiveBayesFit:
    def test_fit_model_creation(self, setup_fit_models):
        model1 = setup_fit_models["model1"]
        model2 = setup_fit_models["model2"]
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        model1.fit(values, "A")
        assert sorted(model1.nodes()) == sorted(["A", "B", "C", "D", "E"])
        assert sorted(model1.edges()) == sorted([("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")])
        assert model1.dependent == "A"
        assert model1.features == {"B", "C", "D", "E"}

        model2.fit(values)
        assert sorted(model1.nodes()) == sorted(["A", "B", "C", "D", "E"])
        assert sorted(model1.edges()) == sorted([("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")])
        assert model2.dependent == "A"
        assert model2.features == {"B", "C", "D", "E"}

    def test_fit_model_creation_exception(self, setup_fit_models):
        model1 = setup_fit_models["model1"]
        model2 = setup_fit_models["model2"]
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        values2 = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 3)), columns=["C", "D", "E"]
        )

        with pytest.raises(ValueError):
            model1.fit(values)
        with pytest.raises(ValueError):
            model1.fit(values2)
        with pytest.raises(ValueError):
            model2.fit(values2, "A")
