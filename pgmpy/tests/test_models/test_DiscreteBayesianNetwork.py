import os

import networkx as nx
import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest

import pgmpy.tests.help_functions as hf
from pgmpy.base import DAG
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import (
    DiscreteFactor,
    JointProbabilityDistribution,
    TabularCPD,
)
from pgmpy.independencies import Independencies
from pgmpy.inference import ApproxInference, BeliefPropagation
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteEM, DiscreteMLE
from pgmpy.sampling import BayesianModelSampling


class TestBaseModelCreation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.G = DiscreteBayesianNetwork()

    def test_class_init_without_data(self):
        assert isinstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = DiscreteBayesianNetwork([("a", "b"), ("b", "c")])
        assert sorted(self.g.nodes()) == ["a", "b", "c"]
        assert hf.recursive_sorted(self.g.edges()) == [["a", "b"], ["b", "c"]]

    def test_class_init_with_data_nonstring(self):
        DiscreteBayesianNetwork([(1, 2), (2, 3)])

    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = DiscreteBayesianNetwork(adj, latents={"a"})
        assert self.graph.latents == set("a")
        assert sorted(self.graph.nodes()) == ["a", "b", "c"]
        assert self.graph.adj["a"]["c"]["weight"] == 3

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = DiscreteBayesianNetwork(adj, latents={"a"})
        assert self.graph.latents == set("a")
        assert sorted(self.graph.nodes()) == ["a", "b", "c"]

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = DiscreteBayesianNetwork(df, latents={0})
        assert self.graph.latents == {0}
        assert sorted(self.graph.nodes()) == [0, 1]
        assert self.graph.adj[0][1]["weight"] == {"weight": 3}

    def test_add_node_string(self):
        self.G.add_node("a")
        assert list(self.G.nodes()) == ["a"]

    def test_add_node_nonstring(self):
        self.G.add_node(1)

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(["a", "b", "c", "d"])
        assert sorted(self.G.nodes()) == ["a", "b", "c", "d"]

    def test_add_nodes_from_non_string(self):
        self.G.add_nodes_from([1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge("d", "e")
        assert sorted(self.G.nodes()) == ["d", "e"]
        assert list(self.G.edges()) == [("d", "e")]
        self.G.add_nodes_from(["a", "b", "c"])
        self.G.add_edge("a", "b")
        assert hf.recursive_sorted(self.G.edges()) == [["a", "b"], ["d", "e"]]

    def test_add_edge_nonstring(self):
        self.G.add_edge(1, 2)

    def test_add_edge_selfloop(self):
        with pytest.raises(ValueError):
            self.G.add_edge("a", "a")

    def test_add_edge_result_cycle(self):
        self.G.add_edges_from([("a", "b"), ("a", "c")])
        with pytest.raises(ValueError):
            self.G.add_edge("c", "a")

    def test_add_edges_from_string(self):
        self.G.add_edges_from([("a", "b"), ("b", "c")])
        assert sorted(self.G.nodes()) == ["a", "b", "c"]
        assert hf.recursive_sorted(self.G.edges()) == [["a", "b"], ["b", "c"]]
        self.G.add_nodes_from(["d", "e", "f"])
        self.G.add_edges_from([("d", "e"), ("e", "f")])
        assert sorted(self.G.nodes()) == ["a", "b", "c", "d", "e", "f"]
        assert hf.recursive_sorted(self.G.edges()) == hf.recursive_sorted(
            [("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")]
        )

    def test_add_edges_from_nonstring(self):
        self.G.add_edges_from([(1, 2), (2, 3)])

    def test_add_edges_from_self_loop(self):
        with pytest.raises(ValueError):
            self.G.add_edges_from([("a", "a")])

    def test_add_edges_from_result_cycle(self):
        with pytest.raises(ValueError):
            self.G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

    def test_update_node_parents_bm_constructor(self):
        self.g = DiscreteBayesianNetwork([("a", "b"), ("b", "c")])
        assert list(self.g.predecessors("a")) == []
        assert list(self.g.predecessors("b")) == ["a"]
        assert list(self.g.predecessors("c")) == ["b"]

    def test_update_node_parents(self):
        self.G.add_nodes_from(["a", "b", "c"])
        self.G.add_edges_from([("a", "b"), ("b", "c")])
        assert list(self.G.predecessors("a")) == []
        assert list(self.G.predecessors("b")) == ["a"]
        assert list(self.G.predecessors("c")) == ["b"]


class TestBayesianNetworkParser:
    def test_from_lavaan(self):
        model_str = "i =~ x1 + x2 + x3"
        model_from_str = DiscreteBayesianNetwork.from_lavaan(string=model_str)
        expected_edges = {("i", "x1"), ("i", "x2"), ("i", "x3")}
        expected_latents = {"i"}
        assert set(model_from_str.edges()) == expected_edges
        assert set(model_from_str.latents) == expected_latents

    def test_from_daggitty(self):
        dag = DiscreteBayesianNetwork.from_dagitty('dag{ bb="0,0,1,1" X [l, pos="-1.228,-1.145"] X-> {Y Z}  Z->A}')
        assert set(dag.edges()) == {("X", "Z"), ("X", "Y"), ("Z", "A")}
        assert set(dag.latents) == {"X"}


class TestBayesianNetworkMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.G = DiscreteBayesianNetwork([("a", "d"), ("b", "d"), ("d", "e"), ("b", "c")])
        self.G1 = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        diff_cpd = TabularCPD("diff", 2, values=[[0.2], [0.8]])
        intel_cpd = TabularCPD("intel", 3, values=[[0.5], [0.3], [0.2]])
        grade_cpd = TabularCPD(
            "grade",
            3,
            values=[
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        self.G1.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        self.G2 = DiscreteBayesianNetwork([("d", "g"), ("g", "l"), ("i", "g"), ("i", "l")])
        self.G3 = DiscreteBayesianNetwork(
            [
                ("Pop", "EC"),
                ("Urb", "EC"),
                ("GDP", "EC"),
                ("EC", "FFEC"),
                ("EC", "REC"),
                ("EC", "EI"),
                ("REC", "CO2"),
                ("REC", "CH4"),
                ("REC", "N2O"),
                ("FFEC", "CO2"),
                ("FFEC", "CH4"),
                ("FFEC", "N2O"),
            ]
        )

    def test_moral_graph(self):
        moral_graph = self.G.moralize()
        assert sorted(moral_graph.nodes()) == ["a", "b", "c", "d", "e"]
        for edge in moral_graph.edges():
            assert edge in [("a", "b"), ("a", "d"), ("b", "c"), ("d", "b"), ("e", "d")] or (edge[1], edge[0]) in [
                ("a", "b"),
                ("a", "d"),
                ("b", "c"),
                ("d", "b"),
                ("e", "d"),
            ]

    def test_to_junction_tree(self):
        model = DiscreteBayesianNetwork([("a", "b"), ("b", "c"), ("d", "c"), ("e", "d"), ("e", "a")])
        model.add_cpds(
            TabularCPD("e", 2, [[0.4], [0.6]]),
            TabularCPD("a", 2, [[0.3, 0.8], [0.7, 0.2]], evidence=["e"], evidence_card=[2]),
            TabularCPD("b", 2, [[0.5, 0.1], [0.5, 0.9]], evidence=["a"], evidence_card=[2]),
            TabularCPD("d", 2, [[0.6, 0.2], [0.4, 0.8]], evidence=["e"], evidence_card=[2]),
            TabularCPD("c", 2, [[0.9, 0.5, 0.4, 0.1], [0.1, 0.5, 0.6, 0.9]], evidence=["b", "d"], evidence_card=[2, 2]),
        )
        default_cliques = {frozenset(c) for c in model.to_junction_tree().nodes()}
        assert frozenset({"a", "b", "e"}) in default_cliques
        ordered_cliques = {frozenset(c) for c in model.to_junction_tree(order=["e", "a", "b", "c", "d"]).nodes()}
        assert frozenset({"a", "d", "e"}) in ordered_cliques
        assert default_cliques != ordered_cliques
        with pytest.raises(ValueError):
            model.to_junction_tree(heuristic="bogus")

    def test_moral_graph_with_edge_present_over_parents(self):
        G = DiscreteBayesianNetwork([("a", "d"), ("d", "e"), ("b", "d"), ("b", "c"), ("a", "b")])
        moral_graph = G.moralize()
        assert sorted(moral_graph.nodes()) == ["a", "b", "c", "d", "e"]
        for edge in moral_graph.edges():
            assert edge in [("a", "b"), ("c", "b"), ("d", "a"), ("d", "b"), ("d", "e")] or (edge[1], edge[0]) in [
                ("a", "b"),
                ("c", "b"),
                ("d", "a"),
                ("d", "b"),
                ("d", "e"),
            ]

    def test_get_ancestors_success(self):
        ancestors1 = self.G2.get_ancestors("g")
        ancestors2 = self.G2.get_ancestors("d")
        ancestors3 = self.G2.get_ancestors(["i", "l"])
        assert ancestors1 == {"d", "i", "g"}
        assert ancestors2 == {"d"}
        assert ancestors3 == {"g", "i", "l", "d"}

    def test_get_ancestors_failure(self):
        with pytest.raises(ValueError):
            self.G2.get_ancestors("h")

    def test_get_cardinality(self):
        assert self.G1.get_cardinality() == {"diff": 2, "intel": 3, "grade": 3}

    def test_states(self):
        assert self.G1.states == {"diff": [0, 1], "intel": [0, 1, 2], "grade": [0, 1, 2]}

    def test_get_cardinality_with_node(self):
        assert self.G1.get_cardinality("diff") == 2
        assert self.G1.get_cardinality("intel") == 3
        assert self.G1.get_cardinality("grade") == 3

    def test_local_independencies(self):
        assert self.G.local_independencies("a") == Independencies(["a", ["b", "c"]])
        assert self.G.local_independencies("c") == Independencies(["c", ["a", "d", "e"], "b"])
        assert self.G.local_independencies("d") == Independencies(["d", "c", ["b", "a"]])
        assert self.G.local_independencies("e") == Independencies(["e", ["c", "b", "a"], "d"])
        assert self.G.local_independencies("b") == Independencies(["b", "a"])
        assert self.G1.local_independencies("grade") == Independencies()

    def test_get_independencies(self):
        chain = DiscreteBayesianNetwork([("X", "Y"), ("Y", "Z")])
        assert chain.get_independencies() == Independencies(("X", "Z", "Y"), ("Z", "X", "Y"))
        fork = DiscreteBayesianNetwork([("Y", "X"), ("Y", "Z")])
        assert fork.get_independencies() == Independencies(("X", "Z", "Y"), ("Z", "X", "Y"))
        collider = DiscreteBayesianNetwork([("X", "Y"), ("Z", "Y")])
        assert collider.get_independencies() == Independencies(("X", "Z"), ("Z", "X"))

        # Latent variables
        fork = DiscreteBayesianNetwork([("Y", "X"), ("Y", "Z")], latents=["Y"])
        assert fork.get_independencies(include_latents=True) == Independencies(("X", "Z", "Y"), ("Z", "X", "Y"))
        assert fork.get_independencies(include_latents=False) == Independencies()

    def test_is_imap(self):
        val = [
            0.01,
            0.01,
            0.08,
            0.006,
            0.006,
            0.048,
            0.004,
            0.004,
            0.032,
            0.04,
            0.04,
            0.32,
            0.024,
            0.024,
            0.192,
            0.016,
            0.016,
            0.128,
        ]
        JPD = JointProbabilityDistribution(["diff", "intel", "grade"], [2, 3, 3], val)
        fac = DiscreteFactor(["diff", "intel", "grade"], [2, 3, 3], val)
        assert self.G1.is_imap(JPD)
        with pytest.raises(TypeError):
            self.G1.is_imap(fac)

    def test_markov_blanket(self):
        G = DAG(
            [
                ("x", "y"),
                ("z", "y"),
                ("y", "w"),
                ("y", "v"),
                ("u", "w"),
                ("s", "v"),
                ("w", "t"),
                ("w", "m"),
                ("v", "n"),
                ("v", "q"),
            ]
        )
        assert set(G.get_markov_blanket("y")) == {"s", "w", "x", "u", "z", "v"}

    def test_markov_blanket_G3(self):
        assert set(self.G3.get_markov_blanket("CH4")) == {"FFEC", "REC"}

    def test_get_immoralities(self):
        G = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("x", "z"), ("w", "y")])
        imm = G.get_immoralities()
        assert imm["x"] == []
        assert imm["z"] == []
        assert imm["w"] == []
        assert sorted(imm["y"]) == sorted([("w", "x"), ("w", "z")])

        G1 = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("z", "x"), ("w", "y")])
        imm = G1.get_immoralities()
        assert imm["x"] == []
        assert imm["z"] == []
        assert imm["w"] == []
        assert sorted(imm["y"]) == sorted([("w", "x"), ("w", "z")])

        G2 = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("x", "z"), ("w", "y"), ("w", "x")])
        imm = G2.get_immoralities()
        assert imm["x"] == []
        assert imm["z"] == []
        assert imm["w"] == []
        assert imm["y"] == [("w", "z")]

    def test_is_iequivalent(self):
        G = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("x", "z"), ("w", "y")])
        with pytest.raises(TypeError):
            G.is_iequivalent(DiscreteMarkovNetwork())
        G1 = DiscreteBayesianNetwork([("V", "W"), ("W", "X"), ("X", "Y"), ("Z", "Y")])
        G2 = DiscreteBayesianNetwork([("W", "V"), ("X", "W"), ("X", "Y"), ("Z", "Y")])
        assert G1.is_iequivalent(G2)
        G3 = DiscreteBayesianNetwork([("W", "V"), ("W", "X"), ("Y", "X"), ("Z", "Y")])
        assert not G3.is_iequivalent(G2)

        # New examples
        G = DAG([("I", "G"), ("I", "S")])
        G1 = DAG([("S", "I"), ("G", "I")])
        G2 = DAG([("I", "S"), ("S", "G")])
        G3 = DAG([("S", "I"), ("G", "I"), ("S", "G")])

        dags = [G, G1, G2, G3]
        for g in dags:
            assert g.is_iequivalent(g)

        for i in range(4):
            for j in range(4):
                if i != j:
                    assert not dags[i].is_iequivalent(dags[j])

        # Example from Issue #1806.
        G1 = DAG([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        G2 = DAG([("B", "A"), ("C", "A"), ("D", "B"), ("D", "C")])
        assert not G1.is_iequivalent(G2)

    def test_copy(self):
        model_copy = self.G1.copy()
        assert sorted(self.G1.nodes()) == sorted(model_copy.nodes())
        assert sorted(self.G1.edges()) == sorted(model_copy.edges())
        assert id(self.G1.get_cpds("diff")) != id(model_copy.get_cpds("diff"))

        self.G1.remove_cpds("diff")
        diff_cpd = TabularCPD("diff", 2, values=[[0.3], [0.7]])
        self.G1.add_cpds(diff_cpd)
        assert self.G1.get_cpds("diff") != model_copy.get_cpds("diff")

        self.G1.remove_node("intel")
        assert sorted(self.G1.nodes()) != sorted(model_copy.nodes())
        assert sorted(self.G1.edges()) != sorted(model_copy.edges())

        self.G1.with_role(role="exposure", variables="diff", inplace=True)
        model_copy = self.G1.copy()
        assert self.G1.exposures == model_copy.exposures

        model_copy.with_role(role="outcome", variables="grade", inplace=True)
        assert "grade" not in self.G1.get_role_dict().get("outcome", [])

        self.G1.latents = {"diff"}
        model_copy = self.G1.copy()
        assert model_copy.latents == {"diff"}
        model_copy.latents.add("intel")
        assert "intel" not in self.G1.latents

    def test_get_random(self):
        model = DiscreteBayesianNetwork.get_random(n_nodes=5, edge_prob=0.5)
        assert len(model.nodes()) == 5
        assert len(model.cpds) == 5
        for cpd in model.cpds:
            assert np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01)

        model = DiscreteBayesianNetwork.get_random(n_nodes=5, n_edges=4, rng=42)
        assert len(model.edges()) == 4
        assert len(model.cpds) == 5

        # With node names
        node_names = ["a", "aa", "aaa", "aaaa", "aaaaa"]
        model = DiscreteBayesianNetwork.get_random(n_nodes=5, edge_prob=0.5, node_names=node_names)
        assert len(model.nodes()) == 5
        assert sorted(model.nodes()) == node_names
        assert len(model.cpds) == 5
        for cpd in model.cpds:
            assert np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01)

        model = DiscreteBayesianNetwork.get_random(n_nodes=5, edge_prob=0.6, n_states=5)
        assert len(model.nodes()) == 5
        assert len(model.cpds) == 5
        for cpd in model.cpds:
            assert np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01)

        # With node names
        model = DiscreteBayesianNetwork.get_random(n_nodes=5, edge_prob=0.6, node_names=node_names, n_states=5)
        assert len(model.nodes()) == 5
        assert sorted(model.nodes()) == node_names
        assert len(model.cpds) == 5
        for cpd in model.cpds:
            assert np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01)

        model = DiscreteBayesianNetwork.get_random(
            n_nodes=5,
            edge_prob=0.6,
            n_states={"X_0": 2, "X_1": 3, "X_2": 4, "X_3": 5, "X_4": 6},
        )
        assert len(model.nodes()) == 5
        assert len(model.cpds) == 5
        for cpd in model.cpds:
            assert np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01)

        # With node names
        model = DiscreteBayesianNetwork.get_random(
            n_nodes=5,
            edge_prob=0.6,
            node_names=node_names,
            n_states={"a": 2, "aa": 3, "aaa": 4, "aaaa": 5, "aaaaa": 6},
        )
        assert len(model.nodes()) == 5
        assert sorted(model.nodes()) == node_names
        assert model.states == {
            "a": [0, 1],
            "aa": [0, 1, 2],
            "aaa": [0, 1, 2, 3],
            "aaaa": [0, 1, 2, 3, 4],
            "aaaaa": [0, 1, 2, 3, 4, 5],
        }
        assert len(model.cpds) == 5
        for cpd in model.cpds:
            assert np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01)

    def test_get_random_cpds(self):
        model = DiscreteBayesianNetwork(DAG.get_random(n_nodes=5, edge_prob=0.5).edges())
        model.add_nodes_from(["X_0", "X_1", "X_2", "X_3", "X_4"])

        cpds = model.get_random_cpds()
        assert len(cpds) == 5

        model.add_cpds(*cpds)
        assert model.check_model()

        cpds = model.get_random_cpds(n_states=4, rng=42)
        assert len(cpds) == 5

        model.add_cpds(*cpds)
        assert model.check_model()
        assert all([card == 4 for var, card in model.get_cardinality().items()])

        n_states_dict = {"X_0": 3, "X_1": 5, "X_2": 4, "X_3": 9, "X_4": 3}
        cpds = model.get_random_cpds(n_states=n_states_dict, rng=42)
        assert len(cpds) == 5

        model.add_cpds(*cpds)
        assert model.check_model()
        for var in range(5):
            assert model.get_cardinality("X_" + str(var)) == n_states_dict["X_" + str(var)]

        model.get_random_cpds(inplace=True, rng=42)
        assert len(model.cpds) == 5
        assert model.check_model()

    def test_remove_node(self):
        self.G1.remove_node("diff")
        assert sorted(self.G1.nodes()) == sorted(["grade", "intel"])
        with pytest.raises(ValueError):
            self.G1.get_cpds("diff")

    def test_remove_nodes_from(self):
        self.G1.remove_nodes_from(["diff", "grade"])
        assert sorted(self.G1.nodes()) == sorted(["intel"])
        with pytest.raises(ValueError):
            self.G1.get_cpds("diff")
        with pytest.raises(ValueError):
            self.G1.get_cpds("grade")

    def test_do(self):
        # One confounder var with treatment T and outcome C: S -> T -> C ; S -> C
        model = DiscreteBayesianNetwork([("S", "T"), ("T", "C"), ("S", "C")])
        cpd_s = TabularCPD(
            variable="S",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"S": ["m", "f"]},
        )
        cpd_t = TabularCPD(
            variable="T",
            variable_card=2,
            values=[[0.25, 0.75], [0.75, 0.25]],
            evidence=["S"],
            evidence_card=[2],
            state_names={"S": ["m", "f"], "T": [0, 1]},
        )
        cpd_c = TabularCPD(
            variable="C",
            variable_card=2,
            values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
            evidence=["S", "T"],
            evidence_card=[2, 2],
            state_names={"S": ["m", "f"], "T": [0, 1], "C": [0, 1]},
        )
        model.add_cpds(cpd_s, cpd_t, cpd_c)

        model_do_inplace = model.do(["T"], inplace=True)
        model_do_new = model.do(["T"], inplace=False)

        for m in [model_do_inplace, model_do_new]:
            assert sorted(list(m.edges())) == sorted([("S", "C"), ("T", "C")])
            assert len(m.cpds) == 3
            np_test.assert_array_equal(m.get_cpds(node="S").values, np.array([0.5, 0.5]))
            np_test.assert_array_equal(m.get_cpds(node="T").values, np.array([0.5, 0.5]))
            np_test.assert_array_equal(
                m.get_cpds(node="C").values,
                np.array([[[0.3, 0.4], [0.7, 0.8]], [[0.7, 0.6], [0.3, 0.2]]]),
            )

        # a tuple is a single (hashable) node, not a collection; the CPD is marginalized to the node itself
        model = DiscreteBayesianNetwork([("X", ("A", 0)), (("A", 0), "Y")])
        model.add_cpds(
            TabularCPD("X", 2, [[0.4], [0.6]]),
            TabularCPD(("A", 0), 2, [[0.2, 0.7], [0.8, 0.3]], evidence=["X"], evidence_card=[2]),
            TabularCPD("Y", 2, [[0.1, 0.9], [0.9, 0.1]], evidence=[("A", 0)], evidence_card=[2]),
        )
        model_do = model.do(("A", 0))
        assert sorted(model_do.edges()) == [(("A", 0), "Y")]
        assert model_do.get_cpds(("A", 0)).variables == [("A", 0)]
        assert model_do.check_model()
        # the original model is untouched
        assert model.get_cpds(("A", 0)).variables == [("A", 0), "X"]

        # without a value the intervened variable's distribution is unspecified: uniform, not the parent-average
        model = DiscreteBayesianNetwork([("S", "T")])
        model.add_cpds(
            TabularCPD("S", 2, [[0.9], [0.1]], state_names={"S": ["m", "f"]}),
            TabularCPD(
                "T",
                2,
                [[0.9, 0.6], [0.1, 0.4]],
                evidence=["S"],
                evidence_card=[2],
                state_names={"S": ["m", "f"], "T": ["no", "yes"]},
            ),
        )
        cpd_t = model.do("T").get_cpds("T")
        np_test.assert_array_equal(cpd_t.values, [0.5, 0.5])
        assert cpd_t.state_names == {"T": ["no", "yes"]}

        # hard intervention do(T = yes): point mass at the given state, edges into T removed
        model_do = model.do({"T": "yes"})
        np_test.assert_array_equal(model_do.get_cpds("T").values, [0.0, 1.0])
        assert model_do.get_cpds("T").state_names == {"T": ["no", "yes"]}
        assert list(model_do.edges()) == []
        assert model_do.check_model()
        assert model.get_cpds("T").variables == ["T", "S"]
        with pytest.raises(ValueError):
            model.do({"T": "maybe"})

        # soft intervention: the CPD of T is replaced by the given distribution; forms can be mixed
        q_t = TabularCPD("T", 2, [[0.2], [0.8]], state_names={"T": ["no", "yes"]})
        model_do = model.do({"S": "m", "T": q_t})
        np_test.assert_array_equal(model_do.get_cpds("T").values, [0.2, 0.8])
        np_test.assert_array_equal(model_do.get_cpds("S").values, [1.0, 0.0])
        assert model_do.check_model()
        # the intervention CPD must be over the variable alone and use the model's states
        with pytest.raises(ValueError):
            model.do({"T": TabularCPD("T", 2, [[0.2, 0.3], [0.8, 0.7]], evidence=["S"], evidence_card=[2])})
        with pytest.raises(ValueError):
            model.do({"T": TabularCPD("T", 2, [[0.2], [0.8]])})
        with pytest.raises(ValueError):
            model.do({"T": TabularCPD("T", 3, [[0.2], [0.3], [0.5]], state_names={"T": ["no", "yes", "maybe"]})})
        # None leaves the distribution unspecified; inplace=True modifies the model itself
        assert model.do({"T": None}, inplace=True) is model
        np_test.assert_array_equal(model.get_cpds("T").values, [0.5, 0.5])
        assert list(model.edges()) == []

    def test_simulate(self):
        asia = load_model("bnlearn/asia")
        n_samples = int(1e3)
        samples = asia.simulate(n_samples=n_samples, show_progress=False)
        assert samples.shape[0] == n_samples

        # Also simulate from a second, different network.
        sachs = load_model("bnlearn/sachs")
        samples = sachs.simulate(n_samples=n_samples, show_progress=False)
        assert samples.shape[0] == n_samples

    def test_simulate_with_partial_samples(self):
        alarm = load_model("bnlearn/alarm")
        partial_cvp = pd.DataFrame(np.random.choice(["LOW", "NORMAL", "HIGH"], int(1e1)), columns=["CVP"])
        samples = alarm.simulate(n_samples=int(1e1), partial_samples=partial_cvp, show_progress=False)
        assert samples.CVP.tolist() == partial_cvp["CVP"].tolist()

    def test_load_save(self):
        test_model_small = load_model("bnlearn/alarm")
        test_model_other = load_model("bnlearn/sachs")
        for model in {test_model_small, test_model_other}:
            for filetype in {"bif", "xmlbif", "xdsl", "net"}:
                model.save("model." + filetype)
                model.save("model.model", filetype=filetype)

                assert os.path.isfile("model." + filetype)
                assert os.path.isfile("model.model")

                read_model1 = DiscreteBayesianNetwork.load("model." + filetype)
                read_model2 = DiscreteBayesianNetwork.load("model.model", filetype=filetype)

                assert set(read_model1.edges()) == set(model.edges())
                assert set(read_model2.edges()) == set(model.edges())
                assert set(read_model1.nodes()) == set(model.nodes())
                assert set(read_model2.nodes()) == set(model.nodes())
                for var in read_model1.nodes():
                    read_cpd = read_model1.get_cpds(var)
                    orig_cpd = model.get_cpds(var)
                    assert read_cpd == orig_cpd

                os.remove("model." + filetype)
                os.remove("model.model")

            with pytest.raises(ValueError):
                model.save("model", filetype=".png")
            with pytest.raises(ValueError):
                model.load("model", filetype=".png")

        # Test for kwarg parameters
        test_model_int_states = DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("C", "D")])
        test_model_int_states.get_random_cpds(inplace=True)
        test_model_int_states.save("model.bif")
        read_model1 = DiscreteBayesianNetwork.load("model.bif", state_name_type=int)
        read_model2 = DiscreteBayesianNetwork.load("model.bif", state_name_type=int)
        assert test_model_int_states.states == read_model1.states
        assert test_model_int_states.states == read_model2.states


class TestBayesianNetworkCPD:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.G = DiscreteBayesianNetwork([("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")])
        self.G2 = DAG([("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")])
        self.G_latent = DAG([("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")], latents=["d", "g"])

    def test_active_trail_nodes(self):
        assert sorted(self.G2.active_trail_nodes("d")["d"]) == ["d", "g", "l"]
        assert sorted(self.G2.active_trail_nodes("i")["i"]) == ["g", "i", "l", "s"]
        assert sorted(self.G2.active_trail_nodes(["d", "i"])["d"]) == ["d", "g", "l"]

        # For model with latent variables
        assert sorted(self.G_latent.active_trail_nodes("d", include_latents=True)["d"]) == ["d", "g", "l"]
        assert sorted(self.G_latent.active_trail_nodes("i", include_latents=True)["i"]) == ["g", "i", "l", "s"]
        assert sorted(self.G_latent.active_trail_nodes(["d", "i"], include_latents=True)["d"]) == ["d", "g", "l"]

        assert sorted(self.G_latent.active_trail_nodes("d", include_latents=False)["d"]) == ["l"]
        assert sorted(self.G_latent.active_trail_nodes("i", include_latents=False)["i"]) == ["i", "l", "s"]
        assert sorted(self.G_latent.active_trail_nodes(["d", "i"], include_latents=False)["d"]) == ["l"]

    def test_active_trail_nodes_args(self):
        assert sorted(self.G2.active_trail_nodes(["d", "l"], observed="g")["d"]) == ["d", "i", "s"]
        assert sorted(self.G2.active_trail_nodes(["d", "l"], observed="g")["l"]) == ["l"]
        assert sorted(self.G2.active_trail_nodes("s", observed=["i", "l"])["s"]) == ["s"]
        assert sorted(self.G2.active_trail_nodes("s", observed=["d", "l"])["s"]) == ["g", "i", "s"]

    def test_is_dconnected_triplets(self):
        assert self.G.is_dconnected("d", "l")
        assert self.G.is_dconnected("g", "s")
        assert not self.G.is_dconnected("d", "i")
        assert self.G.is_dconnected("d", "i", observed="g")
        assert not self.G.is_dconnected("d", "l", observed="g")
        assert not self.G.is_dconnected("i", "l", observed="g")
        assert self.G.is_dconnected("d", "i", observed="l")
        assert not self.G.is_dconnected("g", "s", observed="i")

    def test_is_dconnected(self):
        assert not self.G.is_dconnected("d", "s")
        assert self.G.is_dconnected("s", "l")
        assert self.G.is_dconnected("d", "s", observed="g")
        assert not self.G.is_dconnected("s", "l", observed="g")

    def test_is_dconnected_args(self):
        assert not self.G.is_dconnected("s", "l", "i")
        assert not self.G.is_dconnected("s", "l", "g")
        assert self.G.is_dconnected("d", "s", "l")
        assert not self.G.is_dconnected("d", "s", ["i", "l"])

    def test_get_cpds(self):
        cpd_d = TabularCPD("d", 2, values=np.random.rand(2, 1))
        cpd_i = TabularCPD("i", 2, values=np.random.rand(2, 1))
        cpd_g = TabularCPD(
            "g",
            2,
            values=np.random.rand(2, 4),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )
        cpd_l = TabularCPD("l", 2, values=np.random.rand(2, 2), evidence=["g"], evidence_card=[2])
        cpd_s = TabularCPD("s", 2, values=np.random.rand(2, 2), evidence=["i"], evidence_card=[2])
        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

        assert self.G.get_cpds("d").variable == "d"

    def test_get_cpds1(self):
        self.model = DiscreteBayesianNetwork([("A", "AB")])
        cpd_a = TabularCPD("A", 2, values=np.random.rand(2, 1))
        cpd_ab = TabularCPD("AB", 2, values=np.random.rand(2, 2), evidence=["A"], evidence_card=[2])

        self.model.add_cpds(cpd_a, cpd_ab)
        assert self.model.get_cpds("A").variable == "A"
        assert self.model.get_cpds("AB").variable == "AB"
        with pytest.raises(ValueError):
            self.model.get_cpds("B")

        self.model.add_node("B")
        assert self.model.get_cpds("B") is None

    def test_add_single_cpd(self):
        cpd_s = TabularCPD("s", 2, np.random.rand(2, 2), ["i"], [2])
        self.G.add_cpds(cpd_s)
        assert self.G.get_cpds() == [cpd_s]

    def test_add_multiple_cpds(self):
        cpd_d = TabularCPD("d", 2, values=np.random.rand(2, 1))
        cpd_i = TabularCPD("i", 2, values=np.random.rand(2, 1))
        cpd_g = TabularCPD(
            "g",
            2,
            values=np.random.rand(2, 4),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )
        cpd_l = TabularCPD("l", 2, values=np.random.rand(2, 2), evidence=["g"], evidence_card=[2])
        cpd_s = TabularCPD("s", 2, values=np.random.rand(2, 2), evidence=["i"], evidence_card=[2])

        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
        assert self.G.get_cpds("d") == cpd_d
        assert self.G.get_cpds("i") == cpd_i
        assert self.G.get_cpds("g") == cpd_g
        assert self.G.get_cpds("l") == cpd_l
        assert self.G.get_cpds("s") == cpd_s

    def test_check_model(self):
        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.8, 0.7, 0.6, 0.4]]),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )

        cpd_s = TabularCPD(
            "s",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["i"],
            evidence_card=[2],
        )

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["g"],
            evidence_card=[2],
        )

        self.G.add_cpds(cpd_g, cpd_s, cpd_l)
        with pytest.raises(ValueError):
            self.G.check_model()

        cpd_d = TabularCPD("d", 2, values=[[0.8], [0.2]])
        cpd_i = TabularCPD("i", 2, values=[[0.7], [0.3]])
        self.G.add_cpds(cpd_d, cpd_i)

        assert self.G.check_model()

    def test_check_model1(self):
        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["i"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_g)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_g)

        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.8, 0.7, 0.6, 0.4]]),
            evidence=["d", "s"],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(cpd_g)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_g)

        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["l"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_g)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_g)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["d"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_l)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_l)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.8, 0.7, 0.6, 0.4]]),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(cpd_l)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_l)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array(
                [
                    [0.2, 0.3, 0.4, 0.6, 0.2, 0.3, 0.4, 0.6],
                    [0.8, 0.7, 0.6, 0.4, 0.8, 0.7, 0.6, 0.4],
                ]
            ),
            evidence=["g", "d", "i"],
            evidence_card=[2, 2, 2],
        )
        self.G.add_cpds(cpd_l)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_l)

    def test_check_model2(self):
        cpd_s = TabularCPD(
            "s",
            2,
            values=np.array([[0.5, 0.3], [0.8, 0.7]]),
            evidence=["i"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_s)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_s)

        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.3, 0.7, 0.6, 0.4]]),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(cpd_g)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_g)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3], [0.1, 0.7]]),
            evidence=["g"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_l)
        with pytest.raises(ValueError):
            self.G.check_model()
        self.G.remove_cpds(cpd_l)

    def test_check_model3(self):
        cpd_d = TabularCPD.get_random("d")
        cpd_g = TabularCPD.get_random("g", evidence=["d", "i"])
        cpd_i = TabularCPD.get_random("i")
        cpd_l = TabularCPD.get_random("l", evidence=["g"])
        cpd_s = TabularCPD.get_random("s", evidence=["i"])
        self.G.add_cpds(cpd_d, cpd_g, cpd_i, cpd_l, cpd_s)

        # Check for missing state names for some variables.
        cpd_i.state_names = {}
        self.G.add_cpds(cpd_i)
        with pytest.raises(ValueError):
            self.G.check_model()

        # Check if the cardinality doesn't match between parent and child nodes.
        cpd_i = TabularCPD.get_random("i", cardinality={"i": 3})
        cpd_s = TabularCPD.get_random("s", evidence=["i"], cardinality={"s": 2, "i": 2})
        self.G.add_cpds(cpd_i, cpd_s)
        with pytest.raises(ValueError):
            self.G.check_model()

        # Check if the state names doesn't match between parent and child nodes.
        cpd_i = TabularCPD.get_random("i", state_names={"i": ["i_1", "i_2"]})
        cpd_s = TabularCPD.get_random("s", evidence=["i"], state_names={"i": ["i_3", "i_4"], "s": ["s_1", "s_2"]})
        self.G.add_cpds(cpd_i, cpd_s)
        with pytest.raises(ValueError):
            self.G.check_model()


class TestBayesianNetworkSampleProb:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = load_model("bnlearn/asia")
        self.samples = self.model.simulate(int(1e5), rng=2)
        self.evidence1 = self.samples.iloc[0, :].to_dict()

        self.evidence2 = self.evidence1.copy()
        del self.evidence2["tub"]

        self.evidence3 = self.evidence2.copy()
        del self.evidence3["xray"]

    def test_prob_values(self):
        sample_prob1 = np.round(
            self.samples.loc[
                np.all(
                    self.samples[list(self.evidence1)] == pd.Series(self.evidence1),
                    axis=1,
                )
            ].shape[0]
            / 1e5,
            decimals=2,
        )
        assert sample_prob1 == np.round(self.model.get_state_probability(self.evidence1), decimals=2)

        sample_prob2 = np.round(
            self.samples.loc[
                np.all(
                    self.samples[list(self.evidence2)] == pd.Series(self.evidence2),
                    axis=1,
                )
            ].shape[0]
            / 1e5,
            decimals=2,
        )
        assert sample_prob2 == np.round(self.model.get_state_probability(self.evidence2), decimals=2)

        sample_prob3 = np.round(
            self.samples.loc[
                np.all(
                    self.samples[list(self.evidence3)] == pd.Series(self.evidence3),
                    axis=1,
                )
            ].shape[0]
            / 1e5,
            decimals=2,
        )
        assert sample_prob3 == np.round(self.model.get_state_probability(self.evidence3), decimals=2)


class TestBayesianNetworkFitPredict:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_disconnected = DiscreteBayesianNetwork()
        self.model_disconnected.add_nodes_from(["A", "B", "C", "D", "E"])
        self.model_connected = DiscreteBayesianNetwork([("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")])

        self.model2 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        self.data1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        self.data2 = pd.DataFrame(
            data={
                "A": [0, np.nan, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.nan],
                "D": [np.nan, "Y", np.nan],
            }
        )

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv", dtype=str)
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_bayesian_fit(self):
        self.model2.fit(
            self.data1,
            estimator=DiscreteBayesianEstimator(
                prior_type="dirichlet",
                pseudo_counts={
                    "A": [[9], [3]],
                    "B": [[9], [3]],
                    "C": [[9, 9, 9, 9], [3, 3, 3, 3]],
                },
            ),
        )
        assert self.model2.get_cpds("B") == TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]])

    def test_dag_fit(self):
        model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        data = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        pseudo_counts = {
            "A": [[9], [3]],
            "B": [[9], [3]],
            "C": [[9, 9, 9, 9], [3, 3, 3, 3]],
        }

        fitted_model_bayesian = model.fit(
            data,
            estimator=DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts=pseudo_counts),
        )
        assert fitted_model_bayesian.get_cpds("B") == TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]])

        fitted_model_mle = model.fit(data, estimator=DiscreteMLE())

        assert fitted_model_mle.get_cpds("B") == TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]])

        fitted_model_em = model.fit(data, estimator=DiscreteEM(show_progress=False))

        assert fitted_model_em.get_cpds("B") == TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]])

    def test_fit_update(self):
        model = load_model("bnlearn/asia")
        model_copy = model.copy()
        data = BayesianModelSampling(model).forward_sample(int(1e3))
        model.fit_update(data, n_prev_samples=int(1e3))
        for var in model.nodes():
            assert model_copy.get_cpds(var).__eq__(model.get_cpds(var), atol=0.1)

        model = model_copy.copy()
        model.fit_update(data)
        for var in model.nodes():
            assert model_copy.get_cpds(var).__eq__(model.get_cpds(var), atol=0.1)

    def test_dag_with_independent_node_fit(self):
        edge_list = [("A", "C"), ("B", "C")]
        model = DiscreteBayesianNetwork(edge_list)
        model.add_node("D")
        data = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": [1, 1, 1]})
        pseudo_counts = {
            "A": [[9], [3]],
            "B": [[9], [3]],
            "C": [[9, 9, 9, 9], [3, 3, 3, 3]],
            "D": [[9]],
        }

        fitted_model_bayesian = model.fit(
            data,
            estimator=DiscreteBayesianEstimator(prior_type="dirichlet", pseudo_counts=pseudo_counts),
        )
        assert fitted_model_bayesian.check_model()
        assert sorted(fitted_model_bayesian.nodes()) == ["A", "B", "C", "D"]

    def test_fit_missing_data(self):
        self.model2.fit(self.data2, estimator=DiscreteMLE(state_names={"C": [0, 1]}))
        cpds = {
            TabularCPD("A", 2, [[0.5], [0.5]]),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD(
                "C",
                2,
                [[0, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        }
        assert cpds == set(self.model2.get_cpds())

    def test_disconnected_fit(self):
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        self.model_disconnected.fit(values)

        for node in ["A", "B", "C", "D", "E"]:
            cpd = self.model_disconnected.get_cpds(node)
            assert cpd.variable == node
            np_test.assert_array_equal(cpd.cardinality, np.array([2]))
            value = values.loc[:, node].value_counts() / values.loc[:, node].value_counts().sum()
            value = value.reindex(sorted(value.index)).values
            np_test.assert_array_equal(cpd.values, value)

    def test_predict(self):
        titanic = DiscreteBayesianNetwork()
        titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived")])
        titanic.fit(self.titanic_data2[500:])

        p1_ve = titanic.predict(self.titanic_data2[["Sex", "Pclass"]][:30])

        p1_bp = titanic.predict(self.titanic_data2[["Sex", "Pclass"]][:30], algo=BeliefPropagation)

        assert p1_ve.shape == (30, 3)
        assert p1_bp.shape == (30, 3)
        assert (p1_ve.value_counts() == [10, 9, 3, 3, 3, 2]).all()
        assert (p1_bp.value_counts() == [10, 9, 3, 3, 3, 2]).all()

        p2_ve = titanic.predict(self.titanic_data2[["Survived", "Pclass"]][:30])

        p2_app = titanic.predict(
            self.titanic_data2[["Survived", "Pclass"]][:30],
            algo=ApproxInference,
            rng=42,
        )

        assert p2_ve.shape == (30, 3)
        assert p2_app.shape == (30, 3)
        assert (p2_ve.value_counts() == [12, 7, 4, 4, 2, 1]).all()
        assert (p2_app.value_counts() == [12, 7, 4, 4, 2, 1]).all()

        p3 = titanic.predict(self.titanic_data2[["Survived", "Sex"]][:30])

        gen = np.random.default_rng(seed=42)
        mask = gen.choice([True, False], size=self.titanic_data2[["Survived", "Sex"]][:30].shape)
        p3_nans = titanic.predict(self.titanic_data2[["Survived", "Sex"]][:30].mask(mask))
        assert p3.shape == (30, 3)
        assert p3_nans.shape == (30, 3)
        assert (p3.value_counts() == [12, 12, 3, 3]).all()
        assert (p3_nans.value_counts() == [15, 8, 7]).all()

    def test_predict_stochastic(self):
        titanic = DiscreteBayesianNetwork()
        titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived")])
        titanic.fit(self.titanic_data2[500:])

        p1 = titanic.predict(
            self.titanic_data2[["Sex", "Pclass"]][:30],
            stochastic=True,
            rng=42,
        )
        p2 = titanic.predict(
            self.titanic_data2[["Survived", "Pclass"]][:30],
            stochastic=True,
            rng=42,
        )
        p3 = titanic.predict(
            self.titanic_data2[["Survived", "Sex"]][:30],
            stochastic=True,
            rng=42,
        )

        assert p1.shape == (30, 3)
        assert p1["Survived"].value_counts().loc["0"] == 16
        assert p1["Survived"].value_counts().loc["1"] == 14

        assert p2.shape == (30, 3)
        assert p2["Sex"].value_counts()["male"] == 20
        assert p2["Sex"].value_counts()["female"] == 10

        assert p3.shape == (30, 3)
        assert p3["Pclass"].value_counts().loc["1"] == 8
        assert p3["Pclass"].value_counts().loc["2"] == 5
        assert p3["Pclass"].value_counts().loc["3"] == 17

    def test_connected_predict(self):
        np.random.seed(42)
        values = pd.DataFrame(
            np.array(np.random.randint(low=0, high=2, size=(1000, 5)), dtype=str),
            columns=["A", "B", "C", "D", "E"],
        )
        fit_data = values[:800]
        predict_data = values[800:].copy()
        self.model_connected.fit(fit_data)
        with pytest.raises(ValueError):
            self.model_connected.predict(predict_data)
        predict_data = predict_data.drop("E", axis=1)
        e_predict = self.model_connected.predict(predict_data)
        np_test.assert_array_equal(
            e_predict["E"].values.ravel(),
            np.array(
                [
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
                dtype=str,
            ),
        )

    def test_connected_predict_probability(self):
        np.random.seed(42)
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(100, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        fit_data = values[:80]
        predict_data = values[80:].copy()
        self.model_connected.fit(fit_data)
        predict_data = predict_data.drop("E", axis=1)
        e_prob = self.model_connected.predict_probability(predict_data)
        np_test.assert_allclose(
            e_prob.values.ravel(),
            np.array(
                [
                    0.57894737,
                    0.42105263,
                    0.57894737,
                    0.42105263,
                    0.57894737,
                    0.42105263,
                    0.5,
                    0.5,
                    0.57894737,
                    0.42105263,
                    0.5,
                    0.5,
                    0.57894737,
                    0.42105263,
                    0.57894737,
                    0.42105263,
                    0.57894737,
                    0.42105263,
                    0.5,
                    0.5,
                    0.57894737,
                    0.42105263,
                    0.57894737,
                    0.42105263,
                    0.5,
                    0.5,
                    0.57894737,
                    0.42105263,
                    0.57894737,
                    0.42105263,
                    0.5,
                    0.5,
                    0.57894737,
                    0.42105263,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ]
            ),
            atol=0,
        )
        predict_data = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1, 5)),
            columns=["A", "B", "C", "F", "E"],
        )[:]

    def test_predict_probability_errors(self):
        np.random.seed(42)
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(2, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        fit_data = values[:1]
        predict_data = values[1:].copy()
        self.model_connected.fit(fit_data)
        with pytest.raises(ValueError):
            self.model_connected.predict_probability(predict_data)
        predict_data = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1, 5)),
            columns=["A", "B", "C", "F", "E"],
        )[:]
        with pytest.raises(ValueError):
            self.model_connected.predict_probability(predict_data)

    def test_predict_probability_non_string_nodes(self):
        model = DiscreteBayesianNetwork([(0, 1)])
        cpd0 = TabularCPD(0, 2, [[0.6], [0.4]])
        cpd1 = TabularCPD(1, 2, [[0.8, 0.2], [0.2, 0.8]], evidence=[0], evidence_card=[2])
        model.add_cpds(cpd0, cpd1)

        predict_data = pd.DataFrame({0: [0, 1]})
        result = model.predict_probability(predict_data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["1_0", "1_1"]
        assert len(result) == 2
        np_test.assert_allclose(result["1_0"].values, [0.8, 0.2])
        np_test.assert_allclose(result["1_1"].values, [0.2, 0.8])


class TestDAGCPDOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = DiscreteBayesianNetwork()

    def test_add_single_cpd(self):
        cpd = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd)
        assert self.graph.get_cpds() == [cpd]

    def test_add_multiple_cpds(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        assert self.graph.get_cpds() == [cpd1, cpd2, cpd3]

    def test_remove_single_cpd(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds(cpd1)
        assert self.graph.get_cpds() == [cpd2, cpd3]

    def test_remove_multiple_cpds(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds(cpd1, cpd3)
        assert self.graph.get_cpds() == [cpd2]

    def test_remove_single_cpd_string(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds("diff")
        assert self.graph.get_cpds() == [cpd2, cpd3]

    def test_remove_multiple_cpds_string(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        self.graph.remove_cpds("diff", "grade")
        assert self.graph.get_cpds() == [cpd2]

    def test_get_values_for_node(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        assert self.graph.get_cpds("diff") == cpd1
        assert self.graph.get_cpds("intel") == cpd2
        assert self.graph.get_cpds("grade") == cpd3

    def test_get_values_raises_error(self):
        cpd1 = TabularCPD("diff", 2, values=np.random.rand(2, 1))
        cpd2 = TabularCPD("intel", 2, values=np.random.rand(2, 1))
        cpd3 = TabularCPD(
            "grade",
            2,
            values=np.random.rand(2, 4),
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.graph.add_cpds(cpd1, cpd2, cpd3)
        with pytest.raises(ValueError):
            self.graph.get_cpds("sat")


class TestSimulation:
    @pytest.fixture(autouse=True)
    def setup(self):
        from pgmpy.inference import CausalInference, VariableElimination

        self.alarm = load_model("bnlearn/alarm")
        self.infer_alarm = VariableElimination(self.alarm)
        self.causal_infer_alarm = CausalInference(self.alarm)

        self.con_model = DiscreteBayesianNetwork([("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")])
        cpd_z = TabularCPD("Z", 2, [[0.2], [0.8]])
        cpd_u = TabularCPD("U", 2, [[0.3], [0.7]])
        cpd_x = TabularCPD("X", 2, [[0.1, 0.2, 0.4, 0.7], [0.9, 0.8, 0.6, 0.3]], ["U", "Z"], [2, 2])
        cpd_y = TabularCPD("Y", 2, [[0.2, 0.1, 0.6, 0.4], [0.8, 0.9, 0.4, 0.6]], ["U", "X"], [2, 2])
        self.con_model.add_cpds(cpd_x, cpd_y, cpd_z, cpd_u)
        self.infer_con_model = VariableElimination(self.con_model)
        self.causal_infer_con_model = CausalInference(self.con_model)

    def _test_con_marginals_equal(self, con_model_samples, inference_marginals):
        sample_marginals = {
            node: con_model_samples[node].value_counts() / con_model_samples.shape[0]
            for node in inference_marginals.keys()
        }
        for node in inference_marginals.keys():
            for state in [0, 1]:
                assert np.isclose(
                    inference_marginals[node].get_value(**{node: state}),
                    sample_marginals[node].loc[state],
                    atol=1e-1,
                )

    def _test_alarm_marginals_equal(self, alarm_samples, inference_marginals):
        sample_marginals = {
            node: alarm_samples[node].value_counts() / alarm_samples.shape[0] for node in inference_marginals.keys()
        }
        for node in inference_marginals.keys():
            cpd = self.alarm.get_cpds(node)
            states = cpd.state_names[node]
            for state in states:
                assert np.isclose(
                    inference_marginals[node].get_value(**{node: state}),
                    sample_marginals[node].loc[state],
                    atol=1e-1,
                )

    def test_simulate(self):
        con_model_samples = self.con_model.simulate(n_samples=int(1e4), show_progress=False, rng=42)
        con_inference_marginals = self.infer_con_model.query(self.con_model.nodes(), joint=False)
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        nodes = list(self.alarm.nodes())[:5]
        alarm_samples = self.alarm.simulate(n_samples=int(1e4), show_progress=False, rng=42)
        alarm_inference_marginals = self.infer_alarm.query(list(nodes), joint=False)
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

    def test_simulate_evidence(self):
        con_model_samples = self.con_model.simulate(n_samples=int(1e4), evidence={"U": 1}, show_progress=False, rng=42)
        con_inference_marginals = self.infer_con_model.query(["X", "Y", "Z"], joint=False, evidence={"U": 1})
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        nodes = list(self.alarm.nodes())[:5]
        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4),
            evidence={"MINVOLSET": "HIGH"},
            show_progress=False,
            rng=42,
        )
        alarm_inference_marginals = self.infer_alarm.query(list(nodes), joint=False)
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        with pytest.raises(ValueError):
            self.alarm.simulate(n_samples=int(1e4), evidence={"MINVOLSET": "UNKNOWN"}, rng=42)

    def test_simulate_intervention(self):
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4),
            do={"X": 1},
            show_progress=False,
            rng=42,
        )
        con_inference_marginals = {"Y": self.causal_infer_con_model.query(["Y"], do={"X": 1})}
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4),
            do={"Z": 1},
            show_progress=False,
            rng=42,
        )
        con_inference_marginals = {"X": self.causal_infer_con_model.query(["X"], do={"Z": 1})}
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4),
            do={"CVP": "LOW"},
            show_progress=False,
            rng=42,
        )
        alarm_inference_marginals = {
            "HISTORY": self.causal_infer_alarm.query(["HISTORY"], do={"CVP": "LOW"}),
            "HR": self.causal_infer_alarm.query(["HR"], do={"CVP": "LOW"}),
            "ERRCAUTER": self.causal_infer_alarm.query(["ERRCAUTER"], do={"CVP": "LOW"}),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4),
            do={"MINVOLSET": "NORMAL"},
            show_progress=False,
            rng=42,
        )
        alarm_inference_marginals = {
            "CVP": self.causal_infer_alarm.query(["CVP"], do={"MINVOLSET": "NORMAL"}),
            "HISTORY": self.causal_infer_alarm.query(["HISTORY"], do={"MINVOLSET": "NORMAL"}),
            "HR": self.causal_infer_alarm.query(["HR"], do={"MINVOLSET": "NORMAL"}),
            "ERRCAUTER": self.causal_infer_alarm.query(["ERRCAUTER"], do={"MINVOLSET": "NORMAL"}),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        with pytest.raises(ValueError):
            self.alarm.simulate(n_samples=int(1e4), do={"MINVOLSET": "UNKNOWN"})

    def test_simulate_virtual_evidence(self):
        # Use virtual evidence argument to simulate hard evidence and match values from inference.

        # Simulates hard evidence U = 1
        virtual_evidence = TabularCPD("U", 2, [[0.0], [1.0]])
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), virtual_evidence=[virtual_evidence], show_progress=False
        )
        con_inference_marginals = self.infer_con_model.query(["X", "Y", "Z"], joint=False, evidence={"U": 1})
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        # Simulates hard evidence MINVOLSET=HIGH
        nodes = list(self.alarm.nodes())[:5]
        virtual_evidence = TabularCPD(
            "MINVOLSET",
            3,
            [[0.0], [0.0], [1.0]],
            state_names={"MINVOLSET": ["LOW", "NORMAL", "HIGH"]},
        )
        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4), virtual_evidence=[virtual_evidence], show_progress=False
        )
        alarm_inference_marginals = self.infer_alarm.query(list(nodes), joint=False)
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

    def test_simulate_invalid_virtual_evidence_scope(self):
        model = DiscreteBayesianNetwork([("A", "B")])
        model.add_cpds(
            TabularCPD("A", 2, [[0.5], [0.5]]),
            TabularCPD(
                "B",
                2,
                [[0.7, 0.2], [0.3, 0.8]],
                evidence=["A"],
                evidence_card=[2],
            ),
        )
        bad_virtual_evidence = TabularCPD(
            "B",
            2,
            [[0.6, 0.4], [0.4, 0.6]],
            evidence=["A"],
            evidence_card=[2],
        )
        with pytest.raises(ValueError):
            model.simulate(
                n_samples=1,
                virtual_evidence=[bad_virtual_evidence],
                show_progress=False,
            )

    def test_simulate_virtual_evidence_wrong_cardinality(self):
        bad_virtual_evidence = TabularCPD("U", 3, [[0.2], [0.5], [0.3]])
        with pytest.raises(ValueError):
            self.con_model.simulate(
                n_samples=1,
                virtual_evidence=[bad_virtual_evidence],
                show_progress=False,
            )

    def test_simulate_virtual_evidence_nonexistent_node(self):
        bad_virtual_evidence = TabularCPD("NONEXISTENT", 2, [[0.5], [0.5]])
        with pytest.raises(ValueError):
            self.con_model.simulate(
                n_samples=1,
                virtual_evidence=[bad_virtual_evidence],
                show_progress=False,
            )

    def test_simulate_virtual_intervention(self):
        # Use virtual intervention argument to simulate hard intervention and match values from inference

        # Simulate hard intervention X=1
        virt_inter = TabularCPD("X", 2, [[0.0], [1.0]])
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False
        )
        con_inference_marginals = {
            "Y": self.causal_infer_con_model.query(["Y"], do={"X": 1}),
        }
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        # Simulate hard intervention Z=1
        virt_inter = TabularCPD("Z", 2, [[0.0], [1.0]])
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False
        )
        con_inference_marginals = {"X": self.causal_infer_con_model.query(["X"], do={"Z": 1})}
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        # Simulate hard intervention CVP=LOW
        virt_inter = TabularCPD(
            "CVP",
            3,
            [[1.0], [0.0], [0.0]],
            state_names={"CVP": ["LOW", "NORMAL", "HIGH"]},
        )
        alarm_samples = self.alarm.simulate(n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False)
        alarm_inference_marginals = {
            "HISTORY": self.causal_infer_alarm.query(["HISTORY"], do={"CVP": "LOW"}),
            "HR": self.causal_infer_alarm.query(["HR"], do={"CVP": "LOW"}),
            "ERRCAUTER": self.causal_infer_alarm.query(["ERRCAUTER"], do={"CVP": "LOW"}),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        # Simulate hard intervention MINVOLSET=NORMAL
        virt_inter = TabularCPD(
            "MINVOLSET",
            3,
            [[0.0], [1.0], [0.0]],
            state_names={"MINVOLSET": ["LOW", "NORMAL", "HIGH"]},
        )
        alarm_samples = self.alarm.simulate(n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False)
        alarm_inference_marginals = {
            "CVP": self.causal_infer_alarm.query(["CVP"], do={"MINVOLSET": "NORMAL"}),
            "HISTORY": self.causal_infer_alarm.query(["HISTORY"], do={"MINVOLSET": "NORMAL"}),
            "HR": self.causal_infer_alarm.query(["HR"], do={"MINVOLSET": "NORMAL"}),
            "ERRCAUTER": self.causal_infer_alarm.query(["ERRCAUTER"], do={"MINVOLSET": "NORMAL"}),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        # A genuinely soft intervention: X is drawn from the given distribution and the downstream
        # variables match exact inference on the intervened model; no helper columns leak into the output
        from pgmpy.inference import VariableElimination

        virt_inter = TabularCPD("X", 2, [[0.5], [0.5]])
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False, rng=42
        )
        assert set(con_model_samples.columns) == set(self.con_model.nodes())
        x_marginal = con_model_samples["X"].value_counts(normalize=True)
        assert np.isclose(x_marginal.loc[0], 0.5, atol=0.02)
        soft_model = self.con_model.do({"X": virt_inter})
        con_inference_marginals = VariableElimination(soft_model).query(["Y"], joint=False)
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

    def test_stimulate_missing_mcar(self):
        samples = self.con_model.simulate(n_samples=3000, rng=42)
        assert not samples.isnull().values.any()

        cpd = TabularCPD(
            "Z*",
            2,
            [[0.2], [0.8]],
        )
        samples = self.con_model.simulate(n_samples=3000, missing_prob=cpd, rng=42)
        missing_fraction = samples["Z"].isnull().mean()
        assert missing_fraction >= 0.75
        assert missing_fraction <= 0.85
        assert not samples.drop(columns=["Z"]).isnull().values.any()

        with pytest.raises(ValueError):
            self.con_model.simulate(n_samples=100, missing_prob=0.5)
        with pytest.raises(ValueError):
            cpd = TabularCPD(
                "Z",
                2,
                [[0.5], [0.5]],
            )
            self.con_model.simulate(n_samples=100, missing_prob=cpd)
        with pytest.raises(ValueError):
            cpd = TabularCPD("Z*", 2, [[0.5, 0.5], [0.5, 0.5]], ["A"], [2])
            self.con_model.simulate(n_samples=100, missing_prob=cpd)

        with pytest.raises(ValueError):
            self.con_model.simulate(n_samples=100, missing_prob=[cpd, 0.5])

        with pytest.raises(ValueError):
            cpd = TabularCPD(
                "M*",
                2,
                [[0.5], [0.5]],
            )
            self.con_model.simulate(n_samples=100, missing_prob=cpd)

        with pytest.raises(ValueError):
            cpd = TabularCPD(
                "Z*",
                3,
                [[0.3], [0.5], [0.2]],
            )
            self.con_model.simulate(n_samples=100, missing_prob=cpd)

    def test_stimulate_missing_mnar(self):
        cpd = TabularCPD("Z*", 2, [[0.6, 0.3], [0.4, 0.7]], ["Z"], [2])
        samples = self.con_model.simulate(n_samples=3000, missing_prob=cpd, return_full=True, rng=42)
        missing_fraction_z0 = samples[samples["Z_full"] == 0]["Z"].isnull().mean()
        missing_fraction_z1 = samples[samples["Z_full"] == 1]["Z"].isnull().mean()

        expected_missing_z0 = 0.4
        expected_missing_z1 = 0.7
        assert abs(missing_fraction_z0 - expected_missing_z0) <= 0.1
        assert abs(missing_fraction_z1 - expected_missing_z1) <= 0.1
        assert not samples.drop(columns=["Z"]).isnull().values.any()

    def test_stimulate_missing_mar(self):
        cpd = TabularCPD("Z*", 2, [[0.3, 0.3, 0.4, 0.2], [0.7, 0.7, 0.6, 0.8]], ["X", "Y"], [2, 2])
        samples = self.con_model.simulate(n_samples=3000, missing_prob=cpd, rng=42)
        grouped = samples.groupby(["X", "Y"], observed=False)["Z"]
        for (x, y), group in grouped:
            missing_fraction = group.isnull().mean()
            expected_missing_fraction = cpd.values[1][x, y]
            assert abs(missing_fraction - expected_missing_fraction) <= 0.1
        assert not samples.drop(columns=["Z"]).isnull().values.any()

        # Testing all three missingness at one sampling as list of CPD.
        cpd_1 = TabularCPD("Z*", 2, [[0.3, 0.3, 0.4, 0.2], [0.7, 0.7, 0.6, 0.8]], ["X", "Y"], [2, 2])
        cpd_2 = TabularCPD("Y*", 2, [[0.6, 0.4], [0.4, 0.6]], ["Y"], [2])
        cpd_3 = TabularCPD("U*", 2, [[0.2], [0.8]])
        samples = self.con_model.simulate(n_samples=3000, missing_prob=[cpd_1, cpd_2, cpd_3], return_full=True, rng=42)
        # MAR
        grouped = samples.groupby(["X", "Y"], observed=False)["Z"]
        for (x, z), group in grouped:
            missing_fraction = group.isnull().mean()
            expected_missing_fraction = cpd_1.values[1][int(x), int(z)]
            assert abs(missing_fraction - expected_missing_fraction) <= 0.1

        # MNAR
        missing_fraction_z0 = samples[samples["Y_full"] == 0]["Y"].isnull().mean()
        missing_fraction_z1 = samples[samples["Y_full"] == 1]["Y"].isnull().mean()
        expected_missing_z0 = 0.4
        expected_missing_z1 = 0.6
        assert abs(missing_fraction_z0 - expected_missing_z0) <= 0.1
        assert abs(missing_fraction_z1 - expected_missing_z1) <= 0.1

        # MCAR
        missing_fraction = samples["U"].isnull().mean()
        expected_missing_fraction = 0.8
        assert abs(missing_fraction - expected_missing_fraction) <= 0.1


def test_random_cpds_differ_across_same_shaped_nodes():
    model = DiscreteBayesianNetwork([("A", "C"), ("B", "D")])
    model.get_random_cpds(n_states=2, rng=42, inplace=True)
    assert not np.allclose(model.get_cpds("A").values, model.get_cpds("B").values)
    assert not np.allclose(model.get_cpds("C").values, model.get_cpds("D").values)
    bn = DiscreteBayesianNetwork.get_random(n_nodes=6, edge_prob=0, n_states=2, rng=42)
    roots = [tuple(np.round(bn.get_cpds(node).values.ravel(), 12)) for node in bn.nodes()]
    assert len(set(roots)) == len(roots)


def test_stochastic_predict_draws_fresh_randomness_per_row_group():
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(["X", "E"])
    model.add_cpds(TabularCPD("X", 2, [[0.5], [0.5]]), TabularCPD("E", 2, [[0.5], [0.5]]))
    pred = model.predict(pd.DataFrame({"E": [0] * 50 + [1] * 50}), stochastic=True, rng=0)
    assert pred["X"].iloc[:50].tolist() != pred["X"].iloc[50:].tolist()
