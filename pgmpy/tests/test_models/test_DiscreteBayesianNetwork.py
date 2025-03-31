import os
import unittest

import networkx as nx
import numpy as np
import numpy.testing as np_test
import pandas as pd

import pgmpy.tests.help_functions as hf
from pgmpy.base import DAG
from pgmpy.estimators import (
    BaseEstimator,
    BayesianEstimator,
    MaximumLikelihoodEstimator,
)
from pgmpy.factors.discrete import (
    DiscreteFactor,
    JointProbabilityDistribution,
    TabularCPD,
)
from pgmpy.independencies import Independencies
from pgmpy.inference import ApproxInference, BeliefPropagation
from pgmpy.models import DiscreteBayesianNetwork, MarkovNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TestBaseModelCreation(unittest.TestCase):
    def setUp(self):
        self.G = DiscreteBayesianNetwork()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.G, nx.DiGraph)

    def test_class_init_with_data_string(self):
        self.g = DiscreteBayesianNetwork([("a", "b"), ("b", "c")])
        self.assertListEqual(sorted(self.g.nodes()), ["a", "b", "c"])
        self.assertListEqual(
            hf.recursive_sorted(self.g.edges()), [["a", "b"], ["b", "c"]]
        )

    def test_class_init_with_data_nonstring(self):
        DiscreteBayesianNetwork([(1, 2), (2, 3)])

    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = DiscreteBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertEqual(self.graph.adj["a"]["c"]["weight"], 3)

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = DiscreteBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = DiscreteBayesianNetwork(df, latents=set([0]))
        self.assertEqual(self.graph.latents, set([0]))
        self.assertListEqual(sorted(self.graph.nodes()), [0, 1])
        self.assertEqual(self.graph.adj[0][1]["weight"], {"weight": 3})

    def test_add_node_string(self):
        self.G.add_node("a")
        self.assertListEqual(list(self.G.nodes()), ["a"])

    def test_add_node_nonstring(self):
        self.G.add_node(1)

    def test_add_nodes_from_string(self):
        self.G.add_nodes_from(["a", "b", "c", "d"])
        self.assertListEqual(sorted(self.G.nodes()), ["a", "b", "c", "d"])

    def test_add_nodes_from_non_string(self):
        self.G.add_nodes_from([1, 2, 3, 4])

    def test_add_edge_string(self):
        self.G.add_edge("d", "e")
        self.assertListEqual(sorted(self.G.nodes()), ["d", "e"])
        self.assertListEqual(list(self.G.edges()), [("d", "e")])
        self.G.add_nodes_from(["a", "b", "c"])
        self.G.add_edge("a", "b")
        self.assertListEqual(
            hf.recursive_sorted(self.G.edges()), [["a", "b"], ["d", "e"]]
        )

    def test_add_edge_nonstring(self):
        self.G.add_edge(1, 2)

    def test_add_edge_selfloop(self):
        self.assertRaises(ValueError, self.G.add_edge, "a", "a")

    def test_add_edge_result_cycle(self):
        self.G.add_edges_from([("a", "b"), ("a", "c")])
        self.assertRaises(ValueError, self.G.add_edge, "c", "a")

    def test_add_edges_from_string(self):
        self.G.add_edges_from([("a", "b"), ("b", "c")])
        self.assertListEqual(sorted(self.G.nodes()), ["a", "b", "c"])
        self.assertListEqual(
            hf.recursive_sorted(self.G.edges()), [["a", "b"], ["b", "c"]]
        )
        self.G.add_nodes_from(["d", "e", "f"])
        self.G.add_edges_from([("d", "e"), ("e", "f")])
        self.assertListEqual(sorted(self.G.nodes()), ["a", "b", "c", "d", "e", "f"])
        self.assertListEqual(
            hf.recursive_sorted(self.G.edges()),
            hf.recursive_sorted([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")]),
        )

    def test_add_edges_from_nonstring(self):
        self.G.add_edges_from([(1, 2), (2, 3)])

    def test_add_edges_from_self_loop(self):
        self.assertRaises(ValueError, self.G.add_edges_from, [("a", "a")])

    def test_add_edges_from_result_cycle(self):
        self.assertRaises(
            ValueError, self.G.add_edges_from, [("a", "b"), ("b", "c"), ("c", "a")]
        )

    def test_update_node_parents_bm_constructor(self):
        self.g = DiscreteBayesianNetwork([("a", "b"), ("b", "c")])
        self.assertListEqual(list(self.g.predecessors("a")), [])
        self.assertListEqual(list(self.g.predecessors("b")), ["a"])
        self.assertListEqual(list(self.g.predecessors("c")), ["b"])

    def test_update_node_parents(self):
        self.G.add_nodes_from(["a", "b", "c"])
        self.G.add_edges_from([("a", "b"), ("b", "c")])
        self.assertListEqual(list(self.G.predecessors("a")), [])
        self.assertListEqual(list(self.G.predecessors("b")), ["a"])
        self.assertListEqual(list(self.G.predecessors("c")), ["b"])

    def tearDown(self):
        del self.G


class TestBayesianNetworkParser(unittest.TestCase):
    def test_from_lavaan(self):
        model_str = "i =~ x1 + x2 + x3"
        model_from_str = DiscreteBayesianNetwork.from_lavaan(string=model_str)
        expected_edges = set([("i", "x1"), ("i", "x2"), ("i", "x3")])
        expected_latents = set(["i"])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
        self.assertEqual(set(model_from_str.latents), expected_latents)

    def test_from_daggitty(self):
        dag = DiscreteBayesianNetwork.from_dagitty(
            'dag{ bb="0,0,1,1" X [l, pos="-1.228,-1.145"] X-> {Y Z}  Z->A}'
        )
        self.assertEqual(set(dag.edges()), set([("X", "Z"), ("X", "Y"), ("Z", "A")]))
        self.assertEqual(set(dag.latents), set(["X"]))


class TestBayesianNetworkMethods(unittest.TestCase):
    def setUp(self):
        self.G = DiscreteBayesianNetwork(
            [("a", "d"), ("b", "d"), ("d", "e"), ("b", "c")]
        )
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
        self.G2 = DiscreteBayesianNetwork(
            [("d", "g"), ("g", "l"), ("i", "g"), ("i", "l")]
        )
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
        self.assertListEqual(sorted(moral_graph.nodes()), ["a", "b", "c", "d", "e"])
        for edge in moral_graph.edges():
            self.assertTrue(
                edge in [("a", "b"), ("a", "d"), ("b", "c"), ("d", "b"), ("e", "d")]
                or (edge[1], edge[0])
                in [("a", "b"), ("a", "d"), ("b", "c"), ("d", "b"), ("e", "d")]
            )

    def test_moral_graph_with_edge_present_over_parents(self):
        G = DiscreteBayesianNetwork(
            [("a", "d"), ("d", "e"), ("b", "d"), ("b", "c"), ("a", "b")]
        )
        moral_graph = G.moralize()
        self.assertListEqual(sorted(moral_graph.nodes()), ["a", "b", "c", "d", "e"])
        for edge in moral_graph.edges():
            self.assertTrue(
                edge in [("a", "b"), ("c", "b"), ("d", "a"), ("d", "b"), ("d", "e")]
                or (edge[1], edge[0])
                in [("a", "b"), ("c", "b"), ("d", "a"), ("d", "b"), ("d", "e")]
            )

    def test_get_ancestors_of_success(self):
        ancestors1 = self.G2._get_ancestors_of("g")
        ancestors2 = self.G2._get_ancestors_of("d")
        ancestors3 = self.G2._get_ancestors_of(["i", "l"])
        self.assertEqual(ancestors1, {"d", "i", "g"})
        self.assertEqual(ancestors2, {"d"})
        self.assertEqual(ancestors3, {"g", "i", "l", "d"})

    def test_get_ancestors_of_failure(self):
        self.assertRaises(ValueError, self.G2._get_ancestors_of, "h")

    def test_get_cardinality(self):
        self.assertDictEqual(
            self.G1.get_cardinality(), {"diff": 2, "intel": 3, "grade": 3}
        )

    def test_states(self):
        self.assertDictEqual(
            self.G1.states, {"diff": [0, 1], "intel": [0, 1, 2], "grade": [0, 1, 2]}
        )

    def test_get_cardinality_with_node(self):
        self.assertEqual(self.G1.get_cardinality("diff"), 2)
        self.assertEqual(self.G1.get_cardinality("intel"), 3)
        self.assertEqual(self.G1.get_cardinality("grade"), 3)

    def test_local_independencies(self):
        self.assertEqual(
            self.G.local_independencies("a"), Independencies(["a", ["b", "c"]])
        )
        self.assertEqual(
            self.G.local_independencies("c"),
            Independencies(["c", ["a", "d", "e"], "b"]),
        )
        self.assertEqual(
            self.G.local_independencies("d"), Independencies(["d", "c", ["b", "a"]])
        )
        self.assertEqual(
            self.G.local_independencies("e"),
            Independencies(["e", ["c", "b", "a"], "d"]),
        )
        self.assertEqual(self.G.local_independencies("b"), Independencies(["b", "a"]))
        self.assertEqual(self.G1.local_independencies("grade"), Independencies())

    def test_get_independencies(self):
        chain = DiscreteBayesianNetwork([("X", "Y"), ("Y", "Z")])
        self.assertEqual(
            chain.get_independencies(), Independencies(("X", "Z", "Y"), ("Z", "X", "Y"))
        )
        fork = DiscreteBayesianNetwork([("Y", "X"), ("Y", "Z")])
        self.assertEqual(
            fork.get_independencies(), Independencies(("X", "Z", "Y"), ("Z", "X", "Y"))
        )
        collider = DiscreteBayesianNetwork([("X", "Y"), ("Z", "Y")])
        self.assertEqual(
            collider.get_independencies(), Independencies(("X", "Z"), ("Z", "X"))
        )

        # Latent variables
        fork = DiscreteBayesianNetwork([("Y", "X"), ("Y", "Z")], latents=["Y"])
        self.assertEqual(
            fork.get_independencies(include_latents=True),
            Independencies(("X", "Z", "Y"), ("Z", "X", "Y")),
        )
        self.assertEqual(
            fork.get_independencies(include_latents=False), Independencies()
        )

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
        self.assertTrue(self.G1.is_imap(JPD))
        self.assertRaises(TypeError, self.G1.is_imap, fac)

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
        self.assertEqual(
            set(G.get_markov_blanket("y")), set(["s", "w", "x", "u", "z", "v"])
        )

    def test_markov_blanket_G3(self):
        self.assertEqual(set(self.G3.get_markov_blanket("CH4")), set(["FFEC", "REC"]))

    def test_get_immoralities(self):
        G = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("x", "z"), ("w", "y")])
        imm = G.get_immoralities()
        self.assertEqual(imm["x"], [])
        self.assertEqual(imm["z"], [])
        self.assertEqual(imm["w"], [])
        self.assertEqual(sorted(imm["y"]), sorted([("w", "x"), ("w", "z")]))

        G1 = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("z", "x"), ("w", "y")])
        imm = G1.get_immoralities()
        self.assertEqual(imm["x"], [])
        self.assertEqual(imm["z"], [])
        self.assertEqual(imm["w"], [])
        self.assertEqual(sorted(imm["y"]), sorted([("w", "x"), ("w", "z")]))

        G2 = DiscreteBayesianNetwork(
            [("x", "y"), ("z", "y"), ("x", "z"), ("w", "y"), ("w", "x")]
        )
        imm = G2.get_immoralities()
        self.assertEqual(imm["x"], [])
        self.assertEqual(imm["z"], [])
        self.assertEqual(imm["w"], [])
        self.assertEqual(imm["y"], [("w", "z")])

    def test_is_iequivalent(self):
        G = DiscreteBayesianNetwork([("x", "y"), ("z", "y"), ("x", "z"), ("w", "y")])
        self.assertRaises(TypeError, G.is_iequivalent, MarkovNetwork())
        G1 = DiscreteBayesianNetwork([("V", "W"), ("W", "X"), ("X", "Y"), ("Z", "Y")])
        G2 = DiscreteBayesianNetwork([("W", "V"), ("X", "W"), ("X", "Y"), ("Z", "Y")])
        self.assertTrue(G1.is_iequivalent(G2))
        G3 = DiscreteBayesianNetwork([("W", "V"), ("W", "X"), ("Y", "X"), ("Z", "Y")])
        self.assertFalse(G3.is_iequivalent(G2))

        # New examples
        G = DAG([("I", "G"), ("I", "S")])
        G1 = DAG([("S", "I"), ("G", "I")])
        G2 = DAG([("I", "S"), ("S", "G")])
        G3 = DAG([("S", "I"), ("G", "I"), ("S", "G")])

        dags = [G, G1, G2, G3]
        for g in dags:
            self.assertTrue(g.is_iequivalent(g))

        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertFalse(dags[i].is_iequivalent(dags[j]))

        # Example from Issue #1806.
        G1 = DAG([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        G2 = DAG([("B", "A"), ("C", "A"), ("D", "B"), ("D", "C")])
        self.assertFalse(G1.is_iequivalent(G2))

    def test_copy(self):
        model_copy = self.G1.copy()
        self.assertEqual(sorted(self.G1.nodes()), sorted(model_copy.nodes()))
        self.assertEqual(sorted(self.G1.edges()), sorted(model_copy.edges()))
        self.assertNotEqual(
            id(self.G1.get_cpds("diff")), id(model_copy.get_cpds("diff"))
        )

        self.G1.remove_cpds("diff")
        diff_cpd = TabularCPD("diff", 2, values=[[0.3], [0.7]])
        self.G1.add_cpds(diff_cpd)
        self.assertNotEqual(self.G1.get_cpds("diff"), model_copy.get_cpds("diff"))

        self.G1.remove_node("intel")
        self.assertNotEqual(sorted(self.G1.nodes()), sorted(model_copy.nodes()))
        self.assertNotEqual(sorted(self.G1.edges()), sorted(model_copy.edges()))

    def test_get_random(self):
        model = DiscreteBayesianNetwork.get_random(n_nodes=5, edge_prob=0.5)
        self.assertEqual(len(model.nodes()), 5)
        self.assertEqual(len(model.cpds), 5)
        for cpd in model.cpds:
            self.assertTrue(np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01))

        # With node names
        node_names = ["a", "aa", "aaa", "aaaa", "aaaaa"]
        model = DiscreteBayesianNetwork.get_random(
            n_nodes=5, edge_prob=0.5, node_names=node_names
        )
        self.assertEqual(len(model.nodes()), 5)
        self.assertEqual(sorted(model.nodes()), node_names)
        self.assertEqual(len(model.cpds), 5)
        for cpd in model.cpds:
            self.assertTrue(np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01))

        model = DiscreteBayesianNetwork.get_random(n_nodes=5, edge_prob=0.6, n_states=5)
        self.assertEqual(len(model.nodes()), 5)
        self.assertEqual(len(model.cpds), 5)
        for cpd in model.cpds:
            self.assertTrue(np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01))

        # With node names
        model = DiscreteBayesianNetwork.get_random(
            n_nodes=5, edge_prob=0.6, node_names=node_names, n_states=5
        )
        self.assertEqual(len(model.nodes()), 5)
        self.assertEqual(sorted(model.nodes()), node_names)
        self.assertEqual(len(model.cpds), 5)
        for cpd in model.cpds:
            self.assertTrue(np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01))

        model = DiscreteBayesianNetwork.get_random(
            n_nodes=5,
            edge_prob=0.6,
            n_states={"X_0": 2, "X_1": 3, "X_2": 4, "X_3": 5, "X_4": 6},
        )
        self.assertEqual(len(model.nodes()), 5)
        self.assertEqual(len(model.cpds), 5)
        for cpd in model.cpds:
            self.assertTrue(np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01))

        # With node names
        model = DiscreteBayesianNetwork.get_random(
            n_nodes=5,
            edge_prob=0.6,
            node_names=node_names,
            n_states={"a": 2, "aa": 3, "aaa": 4, "aaaa": 5, "aaaaa": 6},
        )
        self.assertEqual(len(model.nodes()), 5)
        self.assertEqual(sorted(model.nodes()), node_names)
        self.assertEqual(
            model.states,
            {
                "a": [0, 1],
                "aa": [0, 1, 2],
                "aaa": [0, 1, 2, 3],
                "aaaa": [0, 1, 2, 3, 4],
                "aaaaa": [0, 1, 2, 3, 4, 5],
            },
        )
        self.assertEqual(len(model.cpds), 5)
        for cpd in model.cpds:
            self.assertTrue(np.allclose(np.sum(cpd.get_values(), axis=0), 1, atol=0.01))

    def test_get_random_cpds(self):
        model = DiscreteBayesianNetwork(
            DAG.get_random(n_nodes=5, edge_prob=0.5).edges()
        )
        model.add_nodes_from(["X_0", "X_1", "X_2", "X_3", "X_4"])

        cpds = model.get_random_cpds()
        self.assertEqual(len(cpds), 5)

        model.add_cpds(*cpds)
        self.assertTrue(model.check_model())

        cpds = model.get_random_cpds(n_states=4, seed=42)
        self.assertEqual(len(cpds), 5)

        model.add_cpds(*cpds)
        self.assertTrue(model.check_model())
        self.assertTrue(
            all([card == 4 for var, card in model.get_cardinality().items()])
        )

        n_states_dict = {"X_0": 3, "X_1": 5, "X_2": 4, "X_3": 9, "X_4": 3}
        cpds = model.get_random_cpds(n_states=n_states_dict, seed=42)
        self.assertEqual(len(cpds), 5)

        model.add_cpds(*cpds)
        self.assertTrue(model.check_model())
        for var in range(5):
            self.assertEqual(
                model.get_cardinality("X_" + str(var)), n_states_dict["X_" + str(var)]
            )

        model.get_random_cpds(inplace=True, seed=42)
        self.assertEqual(len(model.cpds), 5)
        self.assertTrue(model.check_model())

    def test_remove_node(self):
        self.G1.remove_node("diff")
        self.assertEqual(sorted(self.G1.nodes()), sorted(["grade", "intel"]))
        self.assertRaises(ValueError, self.G1.get_cpds, "diff")

    def test_remove_nodes_from(self):
        self.G1.remove_nodes_from(["diff", "grade"])
        self.assertEqual(sorted(self.G1.nodes()), sorted(["intel"]))
        self.assertRaises(ValueError, self.G1.get_cpds, "diff")
        self.assertRaises(ValueError, self.G1.get_cpds, "grade")

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
            self.assertEqual(sorted(list(m.edges())), sorted([("S", "C"), ("T", "C")]))
            self.assertEqual(len(m.cpds), 3)
            np_test.assert_array_equal(
                m.get_cpds(node="S").values, np.array([0.5, 0.5])
            )
            np_test.assert_array_equal(
                m.get_cpds(node="T").values, np.array([0.5, 0.5])
            )
            np_test.assert_array_equal(
                m.get_cpds(node="C").values,
                np.array([[[0.3, 0.4], [0.7, 0.8]], [[0.7, 0.6], [0.3, 0.2]]]),
            )

    def test_simulate(self):
        asia = get_example_model("asia")
        n_samples = int(1e3)
        samples = asia.simulate(n_samples=n_samples, show_progress=False)
        self.assertEqual(samples.shape[0], n_samples)

        # The probability values don't sum to 1 in this case.
        barley = get_example_model("barley")
        samples = barley.simulate(n_samples=n_samples, show_progress=False)
        self.assertEqual(samples.shape[0], n_samples)

    def test_simulate_with_partial_samples(self):
        alarm = get_example_model("alarm")
        partial_cvp = pd.DataFrame(
            np.random.choice(["LOW", "NORMAL", "HIGH"], int(1e1)), columns=["CVP"]
        )
        samples = alarm.simulate(
            n_samples=int(1e1), partial_samples=partial_cvp, show_progress=False
        )
        self.assertEqual(samples.CVP.tolist(), partial_cvp["CVP"].tolist())

    def test_load_save(self):
        test_model_small = get_example_model("alarm")
        test_model_large = get_example_model("hailfinder")
        for model in {test_model_small, test_model_large}:
            for filetype in {"bif", "xmlbif", "xdsl"}:
                model.save("model." + filetype)
                model.save("model.model", filetype=filetype)

                self.assertTrue(os.path.isfile("model." + filetype))
                self.assertTrue(os.path.isfile("model.model"))

                read_model1 = DiscreteBayesianNetwork.load("model." + filetype)
                read_model2 = DiscreteBayesianNetwork.load(
                    "model.model", filetype=filetype
                )

                self.assertEqual(set(read_model1.edges()), set(model.edges()))
                self.assertEqual(set(read_model2.edges()), set(model.edges()))
                self.assertEqual(set(read_model1.nodes()), set(model.nodes()))
                self.assertEqual(set(read_model2.nodes()), set(model.nodes()))
                for var in read_model1.nodes():
                    read_cpd = read_model1.get_cpds(var)
                    orig_cpd = model.get_cpds(var)
                    self.assertEqual(read_cpd, orig_cpd)

                os.remove("model." + filetype)
                os.remove("model.model")

        # Test for kwarg parameters
        test_model_int_states = DiscreteBayesianNetwork(
            [("A", "B"), ("B", "C"), ("C", "D")]
        )
        test_model_int_states.get_random_cpds(inplace=True)
        test_model_int_states.save("model.bif")
        read_model1 = DiscreteBayesianNetwork.load("model.bif", state_name_type=int)
        read_model2 = DiscreteBayesianNetwork.load(
            "model.bif", n_jobs=1, state_name_type=int
        )
        self.assertTrue(test_model_int_states.states == read_model1.states)
        self.assertTrue(test_model_int_states.states == read_model2.states)

    def tearDown(self):
        del self.G
        del self.G1


class TestBayesianNetworkCPD(unittest.TestCase):
    def setUp(self):
        self.G = DiscreteBayesianNetwork(
            [("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")]
        )
        self.G2 = DAG([("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")])
        self.G_latent = DAG(
            [("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")], latents=["d", "g"]
        )

    def test_active_trail_nodes(self):
        self.assertEqual(sorted(self.G2.active_trail_nodes("d")["d"]), ["d", "g", "l"])
        self.assertEqual(
            sorted(self.G2.active_trail_nodes("i")["i"]), ["g", "i", "l", "s"]
        )
        self.assertEqual(
            sorted(self.G2.active_trail_nodes(["d", "i"])["d"]), ["d", "g", "l"]
        )

        # For model with latent variables
        self.assertEqual(
            sorted(self.G_latent.active_trail_nodes("d", include_latents=True)["d"]),
            ["d", "g", "l"],
        )
        self.assertEqual(
            sorted(self.G_latent.active_trail_nodes("i", include_latents=True)["i"]),
            ["g", "i", "l", "s"],
        )
        self.assertEqual(
            sorted(
                self.G_latent.active_trail_nodes(["d", "i"], include_latents=True)["d"]
            ),
            ["d", "g", "l"],
        )

        self.assertEqual(
            sorted(self.G_latent.active_trail_nodes("d", include_latents=False)["d"]),
            ["l"],
        )
        self.assertEqual(
            sorted(self.G_latent.active_trail_nodes("i", include_latents=False)["i"]),
            ["i", "l", "s"],
        )
        self.assertEqual(
            sorted(
                self.G_latent.active_trail_nodes(["d", "i"], include_latents=False)["d"]
            ),
            ["l"],
        )

    def test_active_trail_nodes_args(self):
        self.assertEqual(
            sorted(self.G2.active_trail_nodes(["d", "l"], observed="g")["d"]),
            ["d", "i", "s"],
        )
        self.assertEqual(
            sorted(self.G2.active_trail_nodes(["d", "l"], observed="g")["l"]), ["l"]
        )
        self.assertEqual(
            sorted(self.G2.active_trail_nodes("s", observed=["i", "l"])["s"]), ["s"]
        )
        self.assertEqual(
            sorted(self.G2.active_trail_nodes("s", observed=["d", "l"])["s"]),
            ["g", "i", "s"],
        )

    def test_is_dconnected_triplets(self):
        self.assertTrue(self.G.is_dconnected("d", "l"))
        self.assertTrue(self.G.is_dconnected("g", "s"))
        self.assertFalse(self.G.is_dconnected("d", "i"))
        self.assertTrue(self.G.is_dconnected("d", "i", observed="g"))
        self.assertFalse(self.G.is_dconnected("d", "l", observed="g"))
        self.assertFalse(self.G.is_dconnected("i", "l", observed="g"))
        self.assertTrue(self.G.is_dconnected("d", "i", observed="l"))
        self.assertFalse(self.G.is_dconnected("g", "s", observed="i"))

    def test_is_dconnected(self):
        self.assertFalse(self.G.is_dconnected("d", "s"))
        self.assertTrue(self.G.is_dconnected("s", "l"))
        self.assertTrue(self.G.is_dconnected("d", "s", observed="g"))
        self.assertFalse(self.G.is_dconnected("s", "l", observed="g"))

    def test_is_dconnected_args(self):
        self.assertFalse(self.G.is_dconnected("s", "l", "i"))
        self.assertFalse(self.G.is_dconnected("s", "l", "g"))
        self.assertTrue(self.G.is_dconnected("d", "s", "l"))
        self.assertFalse(self.G.is_dconnected("d", "s", ["i", "l"]))

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
        cpd_l = TabularCPD(
            "l", 2, values=np.random.rand(2, 2), evidence=["g"], evidence_card=[2]
        )
        cpd_s = TabularCPD(
            "s", 2, values=np.random.rand(2, 2), evidence=["i"], evidence_card=[2]
        )
        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

        self.assertEqual(self.G.get_cpds("d").variable, "d")

    def test_get_cpds1(self):
        self.model = DiscreteBayesianNetwork([("A", "AB")])
        cpd_a = TabularCPD("A", 2, values=np.random.rand(2, 1))
        cpd_ab = TabularCPD(
            "AB", 2, values=np.random.rand(2, 2), evidence=["A"], evidence_card=[2]
        )

        self.model.add_cpds(cpd_a, cpd_ab)
        self.assertEqual(self.model.get_cpds("A").variable, "A")
        self.assertEqual(self.model.get_cpds("AB").variable, "AB")
        self.assertRaises(ValueError, self.model.get_cpds, "B")

        self.model.add_node("B")
        self.assertIsNone(self.model.get_cpds("B"))

    def test_add_single_cpd(self):
        cpd_s = TabularCPD("s", 2, np.random.rand(2, 2), ["i"], [2])
        self.G.add_cpds(cpd_s)
        self.assertListEqual(self.G.get_cpds(), [cpd_s])

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
        cpd_l = TabularCPD(
            "l", 2, values=np.random.rand(2, 2), evidence=["g"], evidence_card=[2]
        )
        cpd_s = TabularCPD(
            "s", 2, values=np.random.rand(2, 2), evidence=["i"], evidence_card=[2]
        )

        self.G.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
        self.assertEqual(self.G.get_cpds("d"), cpd_d)
        self.assertEqual(self.G.get_cpds("i"), cpd_i)
        self.assertEqual(self.G.get_cpds("g"), cpd_g)
        self.assertEqual(self.G.get_cpds("l"), cpd_l)
        self.assertEqual(self.G.get_cpds("s"), cpd_s)

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
        self.assertRaises(ValueError, self.G.check_model)

        cpd_d = TabularCPD("d", 2, values=[[0.8], [0.2]])
        cpd_i = TabularCPD("i", 2, values=[[0.7], [0.3]])
        self.G.add_cpds(cpd_d, cpd_i)

        self.assertTrue(self.G.check_model())

    def test_check_model1(self):
        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["i"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.8, 0.7, 0.6, 0.4]]),
            evidence=["d", "s"],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["l"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3], [0.8, 0.7]]),
            evidence=["d"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_l)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.8, 0.7, 0.6, 0.4]]),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
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
        self.assertRaises(ValueError, self.G.check_model)
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
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_s)

        cpd_g = TabularCPD(
            "g",
            2,
            values=np.array([[0.2, 0.3, 0.4, 0.6], [0.3, 0.7, 0.6, 0.4]]),
            evidence=["d", "i"],
            evidence_card=[2, 2],
        )
        self.G.add_cpds(cpd_g)
        self.assertRaises(ValueError, self.G.check_model)
        self.G.remove_cpds(cpd_g)

        cpd_l = TabularCPD(
            "l",
            2,
            values=np.array([[0.2, 0.3], [0.1, 0.7]]),
            evidence=["g"],
            evidence_card=[2],
        )
        self.G.add_cpds(cpd_l)
        self.assertRaises(ValueError, self.G.check_model)
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
        self.assertRaises(ValueError, self.G.check_model)

        # Check if the cardinality doesn't match between parent and child nodes.
        cpd_i = TabularCPD.get_random("i", cardinality={"i": 3})
        cpd_s = TabularCPD.get_random("s", evidence=["i"], cardinality={"s": 2, "i": 2})
        self.G.add_cpds(cpd_i, cpd_s)
        self.assertRaises(ValueError, self.G.check_model)

        # Check if the state names doesn't match between parent and child nodes.
        cpd_i = TabularCPD.get_random("i", state_names={"i": ["i_1", "i_2"]})
        cpd_s = TabularCPD.get_random(
            "s", evidence=["i"], state_names={"i": ["i_3", "i_4"], "s": ["s_1", "s_2"]}
        )
        self.G.add_cpds(cpd_i, cpd_s)
        self.assertRaises(ValueError, self.G.check_model)

    def tearDown(self):
        del self.G


class TestBayesianNetworkSampleProb(unittest.TestCase):
    def setUp(self):
        self.model = get_example_model("asia")
        self.samples = self.model.simulate(int(1e5), seed=42)
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
        self.assertEqual(
            sample_prob1,
            np.round(self.model.get_state_probability(self.evidence1), decimals=2),
        )

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
        self.assertEqual(
            sample_prob2,
            np.round(self.model.get_state_probability(self.evidence2), decimals=2),
        )

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
        self.assertEqual(
            sample_prob3,
            np.round(self.model.get_state_probability(self.evidence3), decimals=2),
        )


class TestBayesianNetworkFitPredict(unittest.TestCase):
    def setUp(self):
        self.model_disconnected = DiscreteBayesianNetwork()
        self.model_disconnected.add_nodes_from(["A", "B", "C", "D", "E"])
        self.model_connected = DiscreteBayesianNetwork(
            [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        )

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
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv", dtype=str
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_bayesian_fit(self):
        self.model2.fit(
            self.data1,
            estimator=BayesianEstimator,
            prior_type="dirichlet",
            pseudo_counts={
                "A": [[9], [3]],
                "B": [[9], [3]],
                "C": [[9, 9, 9, 9], [3, 3, 3, 3]],
            },
        )
        self.assertEqual(
            self.model2.get_cpds("B"), TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]])
        )

    def test_fit_update(self):
        model = get_example_model("asia")
        model_copy = model.copy()
        data = BayesianModelSampling(model).forward_sample(int(1e3))
        model.fit_update(data, n_prev_samples=int(1e3))
        for var in model.nodes():
            self.assertTrue(
                model_copy.get_cpds(var).__eq__(model.get_cpds(var), atol=0.1)
            )

        model = model_copy.copy()
        model.fit_update(data)
        for var in model.nodes():
            self.assertTrue(
                model_copy.get_cpds(var).__eq__(model.get_cpds(var), atol=0.1)
            )

    def test_fit_missing_data(self):
        self.model2.fit(self.data2, state_names={"C": [0, 1]})
        cpds = set(
            [
                TabularCPD("A", 2, [[0.5], [0.5]]),
                TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
                TabularCPD(
                    "C",
                    2,
                    [[0, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5]],
                    evidence=["A", "B"],
                    evidence_card=[2, 2],
                ),
            ]
        )
        self.assertSetEqual(cpds, set(self.model2.get_cpds()))

    def test_disconnected_fit(self):
        values = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        self.model_disconnected.fit(values)

        for node in ["A", "B", "C", "D", "E"]:
            cpd = self.model_disconnected.get_cpds(node)
            self.assertEqual(cpd.variable, node)
            np_test.assert_array_equal(cpd.cardinality, np.array([2]))
            value = (
                values.loc[:, node].value_counts()
                / values.loc[:, node].value_counts().sum()
            )
            value = value.reindex(sorted(value.index)).values
            np_test.assert_array_equal(cpd.values, value)

    def test_predict(self):
        titanic = DiscreteBayesianNetwork()
        titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived")])
        titanic.fit(self.titanic_data2[500:])

        p1_ve = titanic.predict(self.titanic_data2[["Sex", "Pclass"]][:30])

        p1_bp = titanic.predict(
            self.titanic_data2[["Sex", "Pclass"]][:30], algo=BeliefPropagation
        )

        self.assertEqual(p1_ve.shape, (30, 3))
        self.assertEqual(p1_bp.shape, (30, 3))
        self.assertTrue((p1_ve.value_counts() == [10, 9, 3, 3, 3, 2]).all())
        self.assertTrue((p1_bp.value_counts() == [10, 9, 3, 3, 3, 2]).all())

        p2_ve = titanic.predict(self.titanic_data2[["Survived", "Pclass"]][:30])

        p2_app = titanic.predict(
            self.titanic_data2[["Survived", "Pclass"]][:30],
            algo=ApproxInference,
            seed=42,
        )

        self.assertEqual(p2_ve.shape, (30, 3))
        self.assertEqual(p2_app.shape, (30, 3))
        self.assertTrue((p2_ve.value_counts() == [12, 7, 4, 4, 2, 1]).all())
        self.assertTrue((p2_app.value_counts() == [12, 7, 4, 4, 2, 1]).all())

        p3 = titanic.predict(self.titanic_data2[["Survived", "Sex"]][:30])

        gen = np.random.default_rng(seed=42)
        mask = gen.choice(
            [True, False], size=self.titanic_data2[["Survived", "Sex"]][:30].shape
        )
        p3_nans = titanic.predict(
            self.titanic_data2[["Survived", "Sex"]][:30].mask(mask)
        )
        self.assertEqual(p3.shape, (30, 3))
        self.assertEqual(p3_nans.shape, (30, 3))
        self.assertTrue((p3.value_counts() == [12, 12, 3, 3]).all())
        self.assertTrue((p3_nans.value_counts() == [15, 8, 7]).all())

    def test_predict_stochastic(self):
        titanic = DiscreteBayesianNetwork()
        titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived")])
        titanic.fit(self.titanic_data2[500:])

        p1 = titanic.predict(
            self.titanic_data2[["Sex", "Pclass"]][:30],
            stochastic=True,
            seed=42,
        )
        p2 = titanic.predict(
            self.titanic_data2[["Survived", "Pclass"]][:30],
            stochastic=True,
            seed=42,
        )
        p3 = titanic.predict(
            self.titanic_data2[["Survived", "Sex"]][:30],
            stochastic=True,
            seed=42,
        )

        self.assertEqual(p1.shape, (30, 3))
        self.assertEqual(p1["Survived"].value_counts().loc["0"], 15)
        self.assertEqual(p1["Survived"].value_counts().loc["1"], 15)

        self.assertEqual(p2.shape, (30, 3))
        self.assertEqual(p2["Sex"].value_counts()["male"], 23)
        self.assertEqual(p2["Sex"].value_counts()["female"], 7)

        self.assertEqual(p3.shape, (30, 3))
        self.assertEqual(p3["Pclass"].value_counts().loc["1"], 6)
        self.assertEqual(p3["Pclass"].value_counts().loc["2"], 3)
        self.assertEqual(p3["Pclass"].value_counts().loc["3"], 21)

    def test_connected_predict(self):
        np.random.seed(42)
        values = pd.DataFrame(
            np.array(np.random.randint(low=0, high=2, size=(1000, 5)), dtype=str),
            columns=["A", "B", "C", "D", "E"],
        )
        fit_data = values[:800]
        predict_data = values[800:].copy()
        self.model_connected.fit(fit_data)
        self.assertRaises(ValueError, self.model_connected.predict, predict_data)
        predict_data.drop("E", axis=1, inplace=True)
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
        predict_data.drop("E", axis=1, inplace=True)
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
        self.assertRaises(
            ValueError, self.model_connected.predict_probability, predict_data
        )
        predict_data = pd.DataFrame(
            np.random.randint(low=0, high=2, size=(1, 5)),
            columns=["A", "B", "C", "F", "E"],
        )[:]
        self.assertRaises(
            ValueError, self.model_connected.predict_probability, predict_data
        )

    def tearDown(self):
        del self.model_connected
        del self.model_disconnected


class TestDAGCPDOperations(unittest.TestCase):
    def setUp(self):
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
        self.assertListEqual(self.graph.get_cpds(), [cpd])

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
        self.assertListEqual(self.graph.get_cpds(), [cpd1, cpd2, cpd3])

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
        self.assertListEqual(self.graph.get_cpds(), [cpd2, cpd3])

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
        self.assertListEqual(self.graph.get_cpds(), [cpd2])

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
        self.assertListEqual(self.graph.get_cpds(), [cpd2, cpd3])

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
        self.assertListEqual(self.graph.get_cpds(), [cpd2])

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
        self.assertEqual(self.graph.get_cpds("diff"), cpd1)
        self.assertEqual(self.graph.get_cpds("intel"), cpd2)
        self.assertEqual(self.graph.get_cpds("grade"), cpd3)

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
        self.assertRaises(ValueError, self.graph.get_cpds, "sat")

    def tearDown(self):
        del self.graph


class TestSimulation(unittest.TestCase):
    def setUp(self):
        from pgmpy.inference import CausalInference, VariableElimination

        self.alarm = get_example_model("alarm")
        self.infer_alarm = VariableElimination(self.alarm)
        self.causal_infer_alarm = CausalInference(self.alarm)

        self.con_model = DiscreteBayesianNetwork(
            [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")]
        )
        cpd_z = TabularCPD("Z", 2, [[0.2], [0.8]])
        cpd_u = TabularCPD("U", 2, [[0.3], [0.7]])
        cpd_x = TabularCPD(
            "X", 2, [[0.1, 0.2, 0.4, 0.7], [0.9, 0.8, 0.6, 0.3]], ["U", "Z"], [2, 2]
        )
        cpd_y = TabularCPD(
            "Y", 2, [[0.2, 0.1, 0.6, 0.4], [0.8, 0.9, 0.4, 0.6]], ["U", "X"], [2, 2]
        )
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
                self.assertTrue(
                    np.isclose(
                        inference_marginals[node].get_value(**{node: state}),
                        sample_marginals[node].loc[state],
                        atol=1e-1,
                    )
                )

    def _test_alarm_marginals_equal(self, alarm_samples, inference_marginals):
        sample_marginals = {
            node: alarm_samples[node].value_counts() / alarm_samples.shape[0]
            for node in inference_marginals.keys()
        }
        for node in inference_marginals.keys():
            cpd = self.alarm.get_cpds(node)
            states = cpd.state_names[node]
            for state in states:
                self.assertTrue(
                    np.isclose(
                        inference_marginals[node].get_value(**{node: state}),
                        sample_marginals[node].loc[state],
                        atol=1e-1,
                    )
                )

    def test_simulate(self):
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), show_progress=False, seed=42
        )
        con_inference_marginals = self.infer_con_model.query(
            self.con_model.nodes(), joint=False
        )
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        nodes = list(self.alarm.nodes())[:5]
        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4), show_progress=False, seed=42
        )
        alarm_inference_marginals = self.infer_alarm.query(list(nodes), joint=False)
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

    def test_simulate_evidence(self):
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), evidence={"U": 1}, show_progress=False, seed=42
        )
        con_inference_marginals = self.infer_con_model.query(
            ["X", "Y", "Z"], joint=False, evidence={"U": 1}
        )
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        nodes = list(self.alarm.nodes())[:5]
        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4),
            evidence={"MINVOLSET": "HIGH"},
            show_progress=False,
            seed=42,
        )
        alarm_inference_marginals = self.infer_alarm.query(list(nodes), joint=False)
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        self.assertRaises(
            ValueError,
            self.alarm.simulate,
            n_samples=int(1e4),
            evidence={"MINVOLSET": "UNKNOWN"},
            seed=42,
        )

    def test_simulate_intervention(self):
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4),
            do={"X": 1},
            show_progress=False,
            seed=42,
        )
        con_inference_marginals = {
            "Y": self.causal_infer_con_model.query(["Y"], do={"X": 1})
        }
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4),
            do={"Z": 1},
            show_progress=False,
            seed=42,
        )
        con_inference_marginals = {
            "X": self.causal_infer_con_model.query(["X"], do={"Z": 1})
        }
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4),
            do={"CVP": "LOW"},
            show_progress=False,
            seed=42,
        )
        alarm_inference_marginals = {
            "HISTORY": self.causal_infer_alarm.query(["HISTORY"], do={"CVP": "LOW"}),
            "HR": self.causal_infer_alarm.query(["HR"], do={"CVP": "LOW"}),
            "ERRCAUTER": self.causal_infer_alarm.query(
                ["ERRCAUTER"], do={"CVP": "LOW"}
            ),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4),
            do={"MINVOLSET": "NORMAL"},
            show_progress=False,
            seed=42,
        )
        alarm_inference_marginals = {
            "CVP": self.causal_infer_alarm.query(["CVP"], do={"MINVOLSET": "NORMAL"}),
            "HISTORY": self.causal_infer_alarm.query(
                ["HISTORY"], do={"MINVOLSET": "NORMAL"}
            ),
            "HR": self.causal_infer_alarm.query(["HR"], do={"MINVOLSET": "NORMAL"}),
            "ERRCAUTER": self.causal_infer_alarm.query(
                ["ERRCAUTER"], do={"MINVOLSET": "NORMAL"}
            ),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        self.assertRaises(
            ValueError,
            self.alarm.simulate,
            n_samples=int(1e4),
            do={"MINVOLSET": "UNKNOWN"},
        )

    def test_simulate_virtual_evidence(self):
        # Use virtual evidence argument to simulate hard evidence and match values from inference.

        # Simulates hard evidence U = 1
        virtual_evidence = TabularCPD("U", 2, [[0.0], [1.0]])
        con_model_samples = self.con_model.simulate(
            n_samples=int(1e4), virtual_evidence=[virtual_evidence], show_progress=False
        )
        con_inference_marginals = self.infer_con_model.query(
            ["X", "Y", "Z"], joint=False, evidence={"U": 1}
        )
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
        con_inference_marginals = {
            "X": self.causal_infer_con_model.query(["X"], do={"Z": 1})
        }
        self._test_con_marginals_equal(con_model_samples, con_inference_marginals)

        # Simulate hard intervention CVP=LOW
        virt_inter = TabularCPD(
            "CVP",
            3,
            [[1.0], [0.0], [0.0]],
            state_names={"CVP": ["LOW", "NORMAL", "HIGH"]},
        )
        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False
        )
        alarm_inference_marginals = {
            "HISTORY": self.causal_infer_alarm.query(["HISTORY"], do={"CVP": "LOW"}),
            "HR": self.causal_infer_alarm.query(["HR"], do={"CVP": "LOW"}),
            "ERRCAUTER": self.causal_infer_alarm.query(
                ["ERRCAUTER"], do={"CVP": "LOW"}
            ),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

        # Simulate hard intervention MINVOLSET=NORMAL
        virt_inter = TabularCPD(
            "MINVOLSET",
            3,
            [[0.0], [1.0], [0.0]],
            state_names={"MINVOLSET": ["LOW", "NORMAL", "HIGH"]},
        )
        alarm_samples = self.alarm.simulate(
            n_samples=int(1e4), virtual_intervention=[virt_inter], show_progress=False
        )
        alarm_inference_marginals = {
            "CVP": self.causal_infer_alarm.query(["CVP"], do={"MINVOLSET": "NORMAL"}),
            "HISTORY": self.causal_infer_alarm.query(
                ["HISTORY"], do={"MINVOLSET": "NORMAL"}
            ),
            "HR": self.causal_infer_alarm.query(["HR"], do={"MINVOLSET": "NORMAL"}),
            "ERRCAUTER": self.causal_infer_alarm.query(
                ["ERRCAUTER"], do={"MINVOLSET": "NORMAL"}
            ),
        }
        self._test_alarm_marginals_equal(alarm_samples, alarm_inference_marginals)

    def test_stimulate_missing_mcar(self):
        samples = self.con_model.simulate(n_samples=3000)
        self.assertFalse(samples.isnull().values.any())

        cpd = TabularCPD(
            "Z*",
            2,
            [[0.2], [0.8]],
        )
        samples = self.con_model.simulate(n_samples=3000, missing_prob=cpd)
        missing_fraction = samples["Z"].isnull().mean()
        self.assertGreaterEqual(missing_fraction, 0.75)
        self.assertLessEqual(missing_fraction, 0.85)
        self.assertFalse(samples.drop(columns=["Z"]).isnull().values.any())

        with self.assertRaises(ValueError):
            self.con_model.simulate(n_samples=100, missing_prob=0.5)
        with self.assertRaises(ValueError):
            cpd = TabularCPD(
                "Z",
                2,
                [[0.5], [0.5]],
            )
            self.con_model.simulate(n_samples=100, missing_prob=cpd)
        with self.assertRaises(ValueError):
            cpd = TabularCPD("Z*", 2, [[0.5, 0.5], [0.5, 0.5]], ["A"], [2])
            self.con_model.simulate(n_samples=100, missing_prob=cpd)

        with self.assertRaises(ValueError):
            self.con_model.simulate(n_samples=100, missing_prob=[cpd, 0.5])

        with self.assertRaises(ValueError):
            cpd = TabularCPD(
                "M*",
                2,
                [[0.5], [0.5]],
            )
            self.con_model.simulate(n_samples=100, missing_prob=cpd)

        with self.assertRaises(ValueError):
            cpd = TabularCPD(
                "Z*",
                3,
                [[0.3], [0.5], [0.2]],
            )
            self.con_model.simulate(n_samples=100, missing_prob=cpd)

    def test_stimulate_missing_mnar(self):
        cpd = TabularCPD("Z*", 2, [[0.6, 0.3], [0.4, 0.7]], ["Z"], [2])
        samples = self.con_model.simulate(
            n_samples=3000, missing_prob=cpd, return_full=True
        )
        missing_fraction_z0 = samples[samples["Z_full"] == 0]["Z"].isnull().mean()
        missing_fraction_z1 = samples[samples["Z_full"] == 1]["Z"].isnull().mean()

        expected_missing_z0 = 0.4
        expected_missing_z1 = 0.7
        self.assertAlmostEqual(missing_fraction_z0, expected_missing_z0, delta=0.1)
        self.assertAlmostEqual(missing_fraction_z1, expected_missing_z1, delta=0.1)
        self.assertFalse(samples.drop(columns=["Z"]).isnull().values.any())

    def test_stimulate_missing_mar(self):
        cpd = TabularCPD(
            "Z*", 2, [[0.3, 0.3, 0.4, 0.2], [0.7, 0.7, 0.6, 0.8]], ["X", "Y"], [2, 2]
        )
        samples = self.con_model.simulate(n_samples=3000, missing_prob=cpd)
        grouped = samples.groupby(["X", "Y"], observed=False)["Z"]
        for (x, y), group in grouped:
            missing_fraction = group.isnull().mean()
            expected_missing_fraction = cpd.values[1][x, y]
            self.assertAlmostEqual(
                missing_fraction, expected_missing_fraction, delta=0.1
            )
        self.assertFalse(samples.drop(columns=["Z"]).isnull().values.any())

        # Testing all three missingness at one sampling as list of CPD.
        cpd_1 = TabularCPD(
            "Z*", 2, [[0.3, 0.3, 0.4, 0.2], [0.7, 0.7, 0.6, 0.8]], ["X", "Y"], [2, 2]
        )
        cpd_2 = TabularCPD("Y*", 2, [[0.6, 0.4], [0.4, 0.6]], ["Y"], [2])
        cpd_3 = TabularCPD("U*", 2, [[0.2], [0.8]])
        samples = self.con_model.simulate(
            n_samples=3000, missing_prob=[cpd_1, cpd_2, cpd_3], return_full=True
        )
        # MAR
        grouped = samples.groupby(["X", "Y"], observed=False)["Z"]
        for (x, z), group in grouped:
            missing_fraction = group.isnull().mean()
            expected_missing_fraction = cpd_1.values[1][int(x), int(z)]
            self.assertAlmostEqual(
                missing_fraction, expected_missing_fraction, delta=0.1
            )

        # MNAR
        missing_fraction_z0 = samples[samples["Y_full"] == 0]["Y"].isnull().mean()
        missing_fraction_z1 = samples[samples["Y_full"] == 1]["Y"].isnull().mean()
        expected_missing_z0 = 0.4
        expected_missing_z1 = 0.6
        self.assertAlmostEqual(missing_fraction_z0, expected_missing_z0, delta=0.1)
        self.assertAlmostEqual(missing_fraction_z1, expected_missing_z1, delta=0.1)

        # MCAR
        missing_fraction = samples["U"].isnull().mean()
        expected_missing_fraction = 0.8
        self.assertAlmostEqual(missing_fraction, expected_missing_fraction, delta=0.1)
