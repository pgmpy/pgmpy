#!/usr/bin/env python3

import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import pgmpy.tests.help_functions as hf
from pgmpy.base import DAG, PDAG
from pgmpy.estimators import (
    BayesianEstimator,
    ExpectationMaximization,
    MaximumLikelihoodEstimator,
)
from pgmpy.estimators.CITests import pearsonr
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN


class TestDAGCreation(unittest.TestCase):
    def setUp(self):
        self.graph = DAG()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, DAG)

    def test_class_init_with_data_string(self):
        self.graph = DAG([("a", "b"), ("b", "c")])
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()), [["a", "b"], ["b", "c"]]
        )
        self.assertEqual(self.graph.latents, set())

        self.graph = DAG([("a", "b"), ("b", "c")], latents=["b"])
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()), [["a", "b"], ["b", "c"]]
        )
        self.assertEqual(self.graph.latents, set(["b"]))

    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = DAG(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertEqual(self.graph.adj["a"]["c"]["weight"], 3)

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = DAG(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = DAG(df, latents=set([0]))
        self.assertEqual(self.graph.latents, set([0]))
        self.assertListEqual(sorted(self.graph.nodes()), [0, 1])
        self.assertEqual(self.graph.adj[0][1]["weight"], {"weight": 3})  # None

    def test_variable_name_contains_non_string_adj_matrix(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = DAG(df)
        self.assertEqual(self.graph._variable_name_contains_non_string(), (0, int))

    def test_variable_name_contains_non_string_mixed_types(self):
        self.graph = DAG([("a", "b"), ("b", "c"), ("a", 3.2)])
        self.graph.nodes()
        self.assertEqual(self.graph._variable_name_contains_non_string(), (3.2, float))

    def test_add_node_string(self):
        self.graph = DAG()
        self.graph.add_node("a")
        self.assertListEqual(list(self.graph.nodes()), ["a"])
        self.assertEqual(self.graph.latents, set())

        self.graph = DAG()
        self.graph.add_node("a", latent=True)
        self.assertListEqual(list(self.graph.nodes()), ["a"])
        self.assertEqual(self.graph.latents, set(["a"]))

    def test_add_node_nonstring(self):
        self.graph = DAG()
        self.graph.add_node(1)

        self.graph = DAG()
        self.graph.add_node(1, latent=True)

    def test_add_nodes_from_string(self):
        self.graph = DAG()
        self.graph.add_nodes_from(["a", "b", "c", "d"])
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c", "d"])
        self.assertEqual(self.graph.latents, set())

        self.graph = DAG()
        self.graph.add_nodes_from(["a", "b", "c", "d"], latent=True)
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c", "d"])
        self.assertEqual(self.graph.latents, set(["a", "b", "c", "d"]))

        self.graph = DAG()
        self.graph.add_nodes_from(
            ["a", "b", "c", "d"], latent=[True, False, True, False]
        )
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c", "d"])
        self.assertEqual(self.graph.latents, set(["a", "c"]))

    def test_add_nodes_from_non_string(self):
        self.graph = DAG()
        self.graph.add_nodes_from([1, 2, 3, 4])

        self.graph = DAG()
        self.graph.add_nodes_from([1, 2, 3, 4], latent=True)

        self.graph = DAG()
        self.graph.add_nodes_from([1, 2, 3, 4], latent=[True, False, False, False])

    def test_add_node_weight(self):
        self.graph = DAG()
        self.graph.add_node("weighted_a", weight=0.3)
        self.assertEqual(self.graph.nodes["weighted_a"]["weight"], 0.3)
        self.assertEqual(self.graph.latents, set())

        self.graph = DAG()
        self.graph.add_node("weighted_a", weight=0.3, latent=True)
        self.assertEqual(self.graph.nodes["weighted_a"]["weight"], 0.3)
        self.assertEqual(self.graph.latents, set(["weighted_a"]))

    def test_add_nodes_from_weight(self):
        self.graph = DAG()
        self.graph.add_nodes_from(["weighted_b", "weighted_c"], weights=[0.5, 0.6])
        self.assertEqual(self.graph.nodes["weighted_b"]["weight"], 0.5)
        self.assertEqual(self.graph.nodes["weighted_c"]["weight"], 0.6)

        self.graph = DAG()
        self.graph.add_nodes_from(
            ["weighted_b", "weighted_c"], weights=[0.5, 0.6], latent=True
        )
        self.assertEqual(self.graph.nodes["weighted_b"]["weight"], 0.5)
        self.assertEqual(self.graph.nodes["weighted_c"]["weight"], 0.6)
        self.assertEqual(self.graph.latents, set(["weighted_b", "weighted_c"]))

        self.graph = DAG()
        self.graph.add_nodes_from(
            ["weighted_b", "weighted_c"], weights=[0.5, 0.6], latent=[True, False]
        )
        self.assertEqual(self.graph.nodes["weighted_b"]["weight"], 0.5)
        self.assertEqual(self.graph.nodes["weighted_c"]["weight"], 0.6)
        self.assertEqual(self.graph.latents, set(["weighted_b"]))

        self.graph.add_nodes_from(["e", "f"])
        self.assertEqual(self.graph.nodes["e"]["weight"], None)
        self.assertEqual(self.graph.nodes["f"]["weight"], None)

    def test_add_edge_string(self):
        self.graph.add_edge("d", "e")
        self.assertListEqual(sorted(self.graph.nodes()), ["d", "e"])
        self.assertListEqual(list(self.graph.edges()), [("d", "e")])
        self.graph.add_nodes_from(["a", "b", "c"])
        self.graph.add_edge("a", "b")
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()), [["a", "b"], ["d", "e"]]
        )

    def test_add_edge_nonstring(self):
        self.graph.add_edge(1, 2)

    def test_add_edges_from_string(self):
        self.graph.add_edges_from([("a", "b"), ("b", "c")])
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()), [["a", "b"], ["b", "c"]]
        )
        self.graph.add_nodes_from(["d", "e", "f"])
        self.graph.add_edges_from([("d", "e"), ("e", "f")])
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c", "d", "e", "f"])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()),
            hf.recursive_sorted([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")]),
        )

    def test_add_edges_from_nonstring(self):
        self.graph.add_edges_from([(1, 2), (2, 3)])

    def test_add_edge_weight(self):
        self.graph.add_edge("a", "b", weight=0.3)
        self.assertEqual(self.graph.adj["a"]["b"]["weight"], 0.3)

    def test_add_edges_from_weight(self):
        self.graph.add_edges_from([("b", "c"), ("c", "d")], weights=[0.5, 0.6])
        self.assertEqual(self.graph.adj["b"]["c"]["weight"], 0.5)
        self.assertEqual(self.graph.adj["c"]["d"]["weight"], 0.6)

        self.graph.add_edges_from([("e", "f")])
        self.assertEqual(self.graph.adj["e"]["f"]["weight"], None)

    def test_update_node_parents_bm_constructor(self):
        self.graph = DAG([("a", "b"), ("b", "c")])
        self.assertListEqual(list(self.graph.predecessors("a")), [])
        self.assertListEqual(list(self.graph.predecessors("b")), ["a"])
        self.assertListEqual(list(self.graph.predecessors("c")), ["b"])

    def test_update_node_parents(self):
        self.graph.add_nodes_from(["a", "b", "c"])
        self.graph.add_edges_from([("a", "b"), ("b", "c")])
        self.assertListEqual(list(self.graph.predecessors("a")), [])
        self.assertListEqual(list(self.graph.predecessors("b")), ["a"])
        self.assertListEqual(list(self.graph.predecessors("c")), ["b"])

    def test_get_leaves(self):
        self.graph.add_edges_from(
            [("A", "B"), ("B", "C"), ("B", "D"), ("D", "E"), ("D", "F"), ("A", "G")]
        )
        self.assertEqual(sorted(self.graph.get_leaves()), sorted(["C", "G", "E", "F"]))

    def test_get_roots(self):
        self.graph.add_edges_from(
            [("A", "B"), ("B", "C"), ("B", "D"), ("D", "E"), ("D", "F"), ("A", "G")]
        )
        self.assertEqual(["A"], self.graph.get_roots())
        self.graph.add_edge("H", "G")
        self.assertEqual(sorted(["A", "H"]), sorted(self.graph.get_roots()))

    def test_init_with_cycle(self):
        self.assertRaises(ValueError, DAG, [("a", "a")])
        self.assertRaises(ValueError, DAG, [("a", "b"), ("b", "a")])
        self.assertRaises(ValueError, DAG, [("a", "b"), ("b", "c"), ("c", "a")])

    def test_get_ancestral_graph(self):
        dag = DAG([("A", "C"), ("B", "C"), ("D", "A"), ("D", "B")])
        anc_dag = dag.get_ancestral_graph(["A", "B"])
        self.assertEqual(set(anc_dag.edges()), set([("D", "A"), ("D", "B")]))
        self.assertRaises(ValueError, dag.get_ancestral_graph, ["A", "gibber"])

    def test_to_pdag(self):
        dag = DAG([("X", "Z"), ("Z", "W"), ("Y", "U")])
        pdag = dag.to_pdag()

        # Expected edges in the PDAG
        expected_edges = {
            ("Y", "U"),
            ("U", "Y"),  # Undirected edge between Y and U
            ("Z", "W"),
            ("W", "Z"),  # Undirected edge between Z and W
            ("X", "Z"),
            ("Z", "X"),  # Undirected edge between X and Z
        }

        # Check that all expected edges are present
        self.assertEqual(set(pdag.edges()), expected_edges)

        # Check that the PDAG has the correct number of nodes
        self.assertEqual(set(pdag.nodes()), {"X", "Y", "Z", "W", "U"})

        # Check that there are no latent variables
        self.assertEqual(pdag.latents, set())

    def test_to_pdag_single_edge(self):
        dag = DAG([("X", "Y")])
        pdag = dag.to_pdag()

        # Expected edges in the PDAG
        expected_edges = {("X", "Y"), ("Y", "X")}
        # Check that all expected edges are present
        self.assertEqual(set(pdag.edges()), expected_edges)
        # Check that the PDAG has the correct number of nodes
        self.assertEqual(set(pdag.nodes()), {"X", "Y"})
        # Check that there are no latent variables
        self.assertEqual(pdag.latents, set())

    def test_to_pdag_v_structure(self):
        dag = DAG([("X", "Y"), ("Z", "Y")])
        pdag = dag.to_pdag()

        # Expected edges in the PDAG
        expected_edges = {("X", "Y"), ("Z", "Y")}
        # Check that all expected edges are present
        self.assertEqual(set(pdag.edges()), expected_edges)
        # Check that the PDAG has the correct number of nodes
        self.assertEqual(set(pdag.nodes()), {"X", "Y", "Z"})
        # Check that there are no latent variables
        self.assertEqual(pdag.latents, set())

    def test_to_pdag_multiple_edges_1(self):
        dag = DAG(
            [
                ("Z1", "X"),
                ("Z1", "Z3"),
                ("Z2", "Z3"),
                ("Z2", "Y"),
                ("Z3", "X"),
                ("Z3", "Y"),
                ("X", "W"),
                ("W", "Y"),
            ]
        )
        pdag = dag.to_pdag()

        # Expected edges in the PDAG
        expected_edges = {
            ("Z1", "Z3"),
            ("Z1", "X"),
            ("Z3", "X"),
            ("Z3", "Y"),
            ("Z2", "Z3"),
            ("Z2", "Y"),
            ("X", "W"),
            ("W", "Y"),
        }

        # Check that all expected edges are present
        self.assertEqual(set(pdag.edges()), expected_edges)
        # Check that the PDAG has the correct number of nodes
        self.assertEqual(set(pdag.nodes()), {"Z1", "Z2", "X", "Y", "Z3", "W"})
        # Check that there are no latent variables
        self.assertEqual(pdag.latents, set())

    def test_to_pdag_multiple_edges_2(self):
        dag = DAG([("X", "Y"), ("Z", "Y"), ("Z", "X")])
        pdag = dag.to_pdag()

        # Expected edges in the PDAG
        expected_edges = {
            ("X", "Y"),
            ("Y", "X"),
            ("Z", "Y"),
            ("Y", "Z"),
            ("Z", "X"),
            ("X", "Z"),
        }
        # Check that all expected edges are present
        self.assertEqual(set(pdag.edges()), expected_edges)
        # Check that the PDAG has the correct number of nodes
        self.assertEqual(set(pdag.nodes()), {"X", "Y", "Z"})
        # Check that there are no latent variables
        self.assertEqual(pdag.latents, set())

    def test_minimal_dseparator(self):
        # Without latent variables

        dag1 = DAG([("A", "B"), ("B", "C")])
        self.assertEqual(dag1.minimal_dseparator(start="A", end="C"), {"B"})

        dag2 = DAG([("A", "B"), ("B", "C"), ("C", "D"), ("A", "E"), ("E", "D")])
        self.assertEqual(dag2.minimal_dseparator(start="A", end="D"), {"C", "E"})

        dag3 = DAG(
            [("B", "A"), ("B", "C"), ("A", "D"), ("D", "C"), ("A", "E"), ("C", "E")]
        )
        self.assertEqual(dag3.minimal_dseparator(start="A", end="C"), {"B", "D"})

        # With latent variables

        dag_lat1 = DAG([("A", "B"), ("B", "C")], latents={"B"})
        self.assertIsNone(dag_lat1.minimal_dseparator(start="A", end="C"))

        dag_lat2 = DAG([("A", "D"), ("D", "B"), ("B", "C")], latents={"B"})
        self.assertEqual(dag_lat2.minimal_dseparator(start="A", end="C"), {"D"})

        dag_lat3 = DAG([("A", "B"), ("B", "D"), ("D", "C")], latents={"B"})
        self.assertEqual(dag_lat3.minimal_dseparator(start="A", end="C"), {"D"})

        dag_lat4 = DAG([("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")], latents={"D"})
        self.assertIsNone(dag_lat4.minimal_dseparator(start="A", end="C"))

        dag_lat5 = DAG(
            [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"), ("E", "C")], latents={"E"}
        )
        self.assertEqual(dag_lat5.minimal_dseparator(start="A", end="C"), {"B", "D"})

    def test_to_daft(self):
        dag = DAG([("A", "C"), ("B", "C"), ("D", "A"), ("D", "B")])
        dag.to_daft(node_pos="circular")

        self.assertRaises(ValueError, dag.to_daft, node_pos="abcd")
        self.assertRaises(ValueError, dag.to_daft, node_pos={"A": (0, 2)})
        self.assertRaises(ValueError, dag.to_daft, node_pos=["random"])

        for layout in [
            "circular",
            "kamada_kawai",
            "planar",
            "random",
            "shell",
            "spring",
            #            "spectral", # TODO: Fails for latest networkx
            "spiral",
        ]:
            dag.to_daft(node_pos=layout)
            dag.to_daft(node_pos=layout, pgm_params={"observed_style": "inner"})
            dag.to_daft(
                node_pos=layout,
                edge_params={("A", "C"): {"label": 2}},
                node_params={"A": {"shape": "rectangle"}},
            )

    def test_random_dag(self):
        for i in range(10):
            n_nodes = 8
            edge_prob = np.random.uniform()
            dag = DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob)
            self.assertEqual(len(dag.nodes()), n_nodes)
            self.assertTrue(nx.is_directed_acyclic_graph(dag))
            self.assertTrue(len(dag.latents) == 0)

            node_names = [
                "a",
                "aa",
                "aaa",
                "aaaa",
                "aaaaa",
                "aaaaaa",
                "aaaaaaa",
                "aaaaaaaa",
            ]
            dag = DAG.get_random(
                n_nodes=n_nodes, edge_prob=edge_prob, node_names=node_names
            )
            self.assertEqual(len(dag.nodes()), n_nodes)
            self.assertEqual(sorted(dag.nodes()), node_names)
            self.assertTrue(nx.is_directed_acyclic_graph(dag))
            self.assertTrue(len(dag.latents) == 0)

    def test_dag_fit(self):
        edge_list = [("A", "C"), ("B", "C")]
        for model in [DAG(edge_list), DiscreteBayesianNetwork(edge_list)]:
            data = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
            pseudo_counts = {
                "A": [[9], [3]],
                "B": [[9], [3]],
                "C": [[9, 9, 9, 9], [3, 3, 3, 3]],
            }

            fitted_model_bayesian = model.fit(
                data,
                estimator=BayesianEstimator,
                prior_type="dirichlet",
                pseudo_counts=pseudo_counts,
            )
            self.assertEqual(
                fitted_model_bayesian.get_cpds("B"),
                TabularCPD("B", 2, [[11.0 / 15], [4.0 / 15]]),
            )

            fitted_model_mle = model.fit(data, estimator=MaximumLikelihoodEstimator)

            self.assertEqual(
                fitted_model_mle.get_cpds("B"),
                TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            )

            fitted_model_em = model.fit(data, estimator=ExpectationMaximization)

            self.assertEqual(
                fitted_model_em.get_cpds("B"),
                TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            )

    def test_dag_with_independent_node_fit(self):
        edge_list = [("A", "C"), ("B", "C")]
        dag = DAG(edge_list)
        dag.add_node("D")
        dbn = DiscreteBayesianNetwork(edge_list)
        dbn.add_node("D")
        for model in [dag, dbn]:
            data = pd.DataFrame(
                data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": [1, 1, 1]}
            )
            pseudo_counts = {
                "A": [[9], [3]],
                "B": [[9], [3]],
                "C": [[9, 9, 9, 9], [3, 3, 3, 3]],
                "D": [[9]],
            }

            fitted_model_bayesian = model.fit(
                data,
                estimator=BayesianEstimator,
                prior_type="dirichlet",
                pseudo_counts=pseudo_counts,
            )
            self.assertTrue(fitted_model_bayesian.check_model())
            self.assertEqual(
                sorted(fitted_model_bayesian.nodes()), ["A", "B", "C", "D"]
            )

    def tearDown(self):
        del self.graph

    def test_edge_strength_basic(self):
        """Test basic functionality and numerical values using simulated data from LinearGaussianBN"""
        # Create a linear Gaussian Bayesian network
        linear_model = LGBN([("X", "Y"), ("Z", "Y")])

        # Create CPDs with specific beta values
        x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        y_cpd = LinearGaussianCPD(
            variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"]
        )
        z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)

        # Add CPDs to the model
        linear_model.add_cpds(x_cpd, y_cpd, z_cpd)

        # Simulate data from the model
        data = linear_model.simulate(n_samples=int(1e4))

        # Create DAG and compute edge strengths
        dag = DAG([("X", "Y"), ("Z", "Y")])
        strengths = dag.edge_strength(data)

        # Test return type and structure
        self.assertTrue(isinstance(strengths, dict))
        self.assertEqual(set(strengths.keys()), {("X", "Y"), ("Z", "Y")})
        self.assertTrue(all(isinstance(v, float) for v in strengths.values()))

        # Test that edge strengths match squared Pearson correlation
        xy_corr = pearsonr("X", "Y", ["Z"], data, boolean=False)
        zy_corr = pearsonr("Z", "Y", ["X"], data, boolean=False)

        self.assertAlmostEqual(strengths[("X", "Y")], xy_corr[0] ** 2, places=2)
        self.assertAlmostEqual(strengths[("Z", "Y")], zy_corr[0] ** 2, places=2)

    def test_edge_strength_specific_edge(self):
        """Test computing strength for specific edge using simulated data"""
        # Create a linear Gaussian Bayesian network
        linear_model = LGBN([("X", "Y"), ("Z", "Y")])

        # Create CPDs with specific beta values
        x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        y_cpd = LinearGaussianCPD(
            variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"]
        )
        z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)

        # Add CPDs to the model
        linear_model.add_cpds(x_cpd, y_cpd, z_cpd)

        # Simulate data from the model
        data = linear_model.simulate(n_samples=int(1e4))

        # Create DAG and compute edge strength for specific edge
        dag = DAG([("X", "Y"), ("Z", "Y")])
        strength_xy = dag.edge_strength(data, edges=("X", "Y"))

        # Test structure
        self.assertEqual(set(strength_xy.keys()), {("X", "Y")})

        # Test that edge strength matches squared Pearson correlation
        xy_corr = pearsonr("X", "Y", ["Z"], data, boolean=False)[0]
        self.assertAlmostEqual(strength_xy[("X", "Y")], xy_corr**2, places=2)

    def test_edge_strength_multiple_edges(self):
        """Test computing strength for multiple specific edges using simulated data"""
        # Create a linear Gaussian Bayesian network
        linear_model = LGBN([("X", "Y"), ("Z", "Y")])

        # Create CPDs with specific beta values
        x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        y_cpd = LinearGaussianCPD(
            variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"]
        )
        z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)

        # Add CPDs to the model
        linear_model.add_cpds(x_cpd, y_cpd, z_cpd)

        # Simulate data from the model
        data = linear_model.simulate(n_samples=int(1e4))

        # Create DAG and compute edge strengths for specific edges
        dag = DAG([("X", "Y"), ("Z", "Y")])
        strengths = dag.edge_strength(data, edges=[("X", "Y"), ("Z", "Y")])

        # Test structure
        self.assertEqual(set(strengths.keys()), {("X", "Y"), ("Z", "Y")})

        # Test that edge strengths match squared Pearson correlation
        xy_corr = pearsonr("X", "Y", ["Z"], data, boolean=False)[0]
        zy_corr = pearsonr("Z", "Y", ["X"], data, boolean=False)[0]

        self.assertAlmostEqual(strengths[("X", "Y")], xy_corr**2, places=2)
        self.assertAlmostEqual(strengths[("Z", "Y")], zy_corr**2, places=2)

    def test_edge_strength_stored_in_graph(self):
        """Test that edge strengths are stored in the graph after computation using simulated data"""
        # Create a linear Gaussian Bayesian network
        linear_model = LGBN([("X", "Y"), ("Z", "Y")])

        # Create CPDs with specific beta values
        x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        y_cpd = LinearGaussianCPD(
            variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"]
        )
        z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)

        # Add CPDs to the model
        linear_model.add_cpds(x_cpd, y_cpd, z_cpd)

        # Simulate data from the model
        data = linear_model.simulate(n_samples=int(1e4))

        # Create DAG and compute edge strengths
        dag = DAG([("X", "Y"), ("Z", "Y")])
        strengths = dag.edge_strength(data)

        # Verify strengths are stored in graph edges
        self.assertIn("strength", dag.edges[("X", "Y")])
        self.assertIn("strength", dag.edges[("Z", "Y")])

        # Verify stored values match computed values
        self.assertAlmostEqual(
            dag.edges[("X", "Y")]["strength"], strengths[("X", "Y")], places=2
        )
        self.assertAlmostEqual(
            dag.edges[("Z", "Y")]["strength"], strengths[("Z", "Y")], places=2
        )

        # Verify stored values match squared Pearson correlation
        xy_corr = pearsonr("X", "Y", ["Z"], data, boolean=False)[0]
        zy_corr = pearsonr("Z", "Y", ["X"], data, boolean=False)[0]

        self.assertAlmostEqual(dag.edges[("X", "Y")]["strength"], xy_corr**2, places=2)
        self.assertAlmostEqual(dag.edges[("Z", "Y")]["strength"], zy_corr**2, places=2)

    def test_edge_strength_invalid_edges(self):
        """Test error handling for invalid edges parameter formats"""
        dag = DAG([("X", "Y"), ("Z", "Y")])
        data = pd.DataFrame({"X": [0, 1, 0, 1], "Y": [1, 3, 0, 2], "Z": [1, 1, 0, 0]})

        # Test invalid single edge format (3-tuple)
        with self.assertRaises(ValueError) as context:
            dag.edge_strength(data, edges=("X", "Y", "extra"))
        self.assertIn(
            "edges parameter must be either None, a 2-tuple (X, Y), or a list of 2-tuples",
            str(context.exception),
        )

        # Test invalid list format (contains non-tuple)
        with self.assertRaises(ValueError) as context:
            dag.edge_strength(data, edges=[("X", "Y"), "invalid"])
        self.assertIn(
            "edges parameter must be either None, a 2-tuple (X, Y), or a list of 2-tuples",
            str(context.exception),
        )

        # Test invalid list format (contains 3-tuple)
        with self.assertRaises(ValueError) as context:
            dag.edge_strength(data, edges=[("X", "Y"), ("Z", "Y", "extra")])
        self.assertIn(
            "edges parameter must be either None, a 2-tuple (X, Y), or a list of 2-tuples",
            str(context.exception),
        )

    def test_edge_strength_skip_latent_edges(self):
        """Test that edge_strength skips edges with latent variables and continues with others"""
        # Create DAG with some latent variables
        dag = DAG([("X", "Y"), ("Z", "Y"), ("L", "X"), ("W", "Z")], latents={"L"})

        # Generate more samples with controlled relationships
        np.random.seed(42)
        n_samples = 100

        # Generate data with some controlled relationships
        data = pd.DataFrame(
            {
                "W": np.random.normal(0, 1, n_samples),
                "L": np.random.normal(0, 1, n_samples),
                "X": np.random.normal(0, 1, n_samples)
                + 0.5 * np.random.normal(0, 1, n_samples),  # X depends on L
                "Z": np.random.normal(0, 1, n_samples)
                + 0.3 * np.random.normal(0, 1, n_samples),  # Z depends on W
                "Y": np.random.normal(0, 1, n_samples)
                + 0.4 * np.random.normal(0, 1, n_samples)
                + 0.3 * np.random.normal(0, 1, n_samples),  # Y depends on X and Z
            }
        )

        # Compute strengths for all edges
        strengths = dag.edge_strength(data)

        # Verify that edges involving latent variables are not in the results
        self.assertNotIn(("L", "X"), strengths)

        # Verify that other edges are computed
        self.assertIn(("X", "Y"), strengths)
        self.assertIn(("Z", "Y"), strengths)
        self.assertIn(("W", "Z"), strengths)

        # Verify that the computed strengths are valid
        for edge in strengths:
            self.assertTrue(0 <= strengths[edge] <= 1)

        # Test with specific edges list
        strengths = dag.edge_strength(data, edges=[("L", "X"), ("X", "Y"), ("W", "Z")])

        # Verify that latent edge is skipped but others are computed
        self.assertNotIn(("L", "X"), strengths)
        self.assertIn(("X", "Y"), strengths)
        self.assertIn(("W", "Z"), strengths)

    def test_edge_strength_plotting_to_daft(self):
        """Test edge strength plotting in to_daft method"""
        dag = DAG([("A", "B"), ("C", "B")])

        with self.assertRaises(ValueError) as context:
            dag.to_daft(plot_edge_strength=True)
        self.assertIn(
            "Edge strength plotting requested but strengths not found",
            str(context.exception),
        )

        dag.edges[("A", "B")]["strength"] = 0.123
        dag.edges[("C", "B")]["strength"] = 0.456

        daft_plot = dag.to_daft(plot_edge_strength=True)
        self.assertIsNotNone(daft_plot)

        dag_no_strength = DAG([("A", "B"), ("C", "B")])
        daft_plot_default = dag_no_strength.to_daft()
        self.assertIsNotNone(daft_plot_default)

    def test_edge_strength_plotting_with_existing_labels(self):
        """Test edge strength plotting when user provides custom edge labels"""
        dag = DAG([("A", "B")])
        dag.edges[("A", "B")]["strength"] = 0.789

        daft_plot = dag.to_daft(
            plot_edge_strength=True, edge_params={("A", "B"): {"label": "custom"}}
        )
        self.assertIsNotNone(daft_plot)


class TestDAGParser(unittest.TestCase):
    def test_from_lavaan(self):
        model_str = """# %load model.lav
                       # measurement model
                         ind60 =~ x1 + x2 + x3
                         dem60 =~ y1 + y2 + y3 + y4
                         dem65 =~ y5 + y6 + y7 + y8
                       # regressions
                         dem60 ~ ind60
                         dem65 ~ ind60 + dem60
                       """
        model_from_str = DAG.from_lavaan(string=model_str)

        with open("test_model.lav", "w") as f:
            f.write(model_str)
        model_from_file = DAG.from_lavaan(filename="test_model.lav")
        os.remove("test_model.lav")

        expected_edges = set(
            [
                ("ind60", "x1"),
                ("ind60", "x2"),
                ("ind60", "x3"),
                ("ind60", "dem60"),
                ("ind60", "dem65"),
                ("dem60", "dem65"),
                ("dem60", "y1"),
                ("dem60", "y2"),
                ("dem60", "y3"),
                ("dem60", "y4"),
                ("dem65", "y5"),
                ("dem65", "y6"),
                ("dem65", "y7"),
                ("dem65", "y8"),
            ]
        )

        expected_latents = set(["dem60", "dem65", "ind60"])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
        self.assertEqual(set(model_from_file.edges()), expected_edges)
        self.assertEqual(set(model_from_str.latents), expected_latents)
        self.assertEqual(set(model_from_file.latents), expected_latents)

    def test_from_lavaan_with_residual_correlation(self):
        model_str = """# %load model_with_residual_correlation.lav
                       # measurement model
                         ind60 =~ x1 + x2 + x3
                       # regressions
                         dem60 ~ ind60
                       # residual correlations
                         y1 ~~ y5
                       """

        model_from_str = DAG.from_lavaan(string=model_str)
        expected_edges = set(
            [
                ("ind60", "x1"),
                ("ind60", "x2"),
                ("ind60", "x3"),
                ("ind60", "dem60"),
            ]
        )

        expected_latents = set(["ind60"])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
        self.assertEqual(set(model_from_str.latents), expected_latents)

    def test_from_dagitty(self):
        model_str = """
            dag{
                smoking "carry matches" [e] ; cancer [o]
                smoking -> {"carry matches" -> cancer} smoking <-> coffee
            }"""
        model_from_str = DAG.from_dagitty(string=model_str)

        with open("test_model.dagitty", "w") as f:
            f.write(model_str)
        model_from_file = DAG.from_dagitty(filename="test_model.dagitty")
        os.remove("test_model.dagitty")

        expected_edges = set(
            [
                ("smoking", "cancer"),
                ("smoking", "carry matches"),
                ("carry matches", "cancer"),
                ("u_coffee_smoking", "coffee"),
                ("u_coffee_smoking", "smoking"),
            ]
        )

        expected_latents = set(["u_coffee_smoking"])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
        self.assertEqual(set(model_from_file.edges()), expected_edges)
        self.assertEqual(set(model_from_str.latents), expected_latents)
        self.assertEqual(set(model_from_file.latents), expected_latents)

    def test_from_dagitty_isolated_nodes(self):
        dag1 = DAG.from_dagitty("dag { A -> B C D -> E F G H} ")
        dag2 = DAG.from_dagitty("dag { A }")
        self.assertEqual(
            set(dag1.nodes()), set(["A", "B", "C", "D", "E", "F", "G", "H"])
        )
        self.assertEqual(set(dag2.nodes()), set(["A"]))
        self.assertEqual(
            set(dag1.edges()),
            set([("A", "B"), ("D", "E")]),
        )
        self.assertEqual(
            set(dag2.edges()),
            set([]),
        )

    def test_from_daggitty_single_line_with_group_of_vars(self):
        dag = DAG.from_dagitty(
            'dag{ bb="0,0,1,1" X [l, pos="-1.228,-1.145"] X-> {Y Z}  Z ->A ->B <- C}'
        )
        self.assertEqual(
            set(dag.edges()),
            set([("X", "Z"), ("X", "Y"), ("Z", "A"), ("A", "B"), ("C", "B")]),
        )
        self.assertEqual(set(dag.latents), set(["X"]))

    def test_from_dagitty_multiline_with_display_info(self):
        dag = DAG.from_dagitty(
            """
                dag {
                bb="-1.728,-4.67,2.587,4.156"
                123 [pos="2.087,3.420"]
                X.1 [pos="-1.228,-1.145"]
                Y [pos="-0.725,-3.934"]
                Z [latent, pos="-0.135,1.659"]
                X.1 -> Y [pos="-0.300,-0.082"]
                X.1 -> Z
                Z -> 123
                }
        """
        )
        self.assertEqual(
            set(dag.edges()), set([("X.1", "Y"), ("X.1", "Z"), ("Z", "123")])
        )
        self.assertEqual(set(dag.latents), set(["Z"]))


class TestDAGMoralization(unittest.TestCase):
    def setUp(self):
        self.graph = DAG()
        self.graph.add_edges_from([("diff", "grade"), ("intel", "grade")])

    def test_get_parents(self):
        self.assertListEqual(sorted(self.graph.get_parents("grade")), ["diff", "intel"])

    def test_moralize(self):
        moral_graph = self.graph.moralize()
        self.assertListEqual(
            hf.recursive_sorted(moral_graph.edges()),
            [["diff", "grade"], ["diff", "intel"], ["grade", "intel"]],
        )

    def test_moralize_disconnected(self):
        graph_copy = self.graph.copy()
        graph_copy.add_node("disconnected")
        moral_graph = graph_copy.moralize()
        self.assertListEqual(
            hf.recursive_sorted(moral_graph.edges()),
            [["diff", "grade"], ["diff", "intel"], ["grade", "intel"]],
        )
        self.assertEqual(
            sorted(moral_graph.nodes()), ["diff", "disconnected", "grade", "intel"]
        )

    def test_get_children(self):
        self.assertListEqual(sorted(self.graph.get_children("diff")), ["grade"])

    def tearDown(self):
        del self.graph


class TestDoOperator(unittest.TestCase):
    def setUp(self):
        self.g1 = DAG([("X", "A"), ("A", "Y"), ("A", "B")])
        self.g2 = DAG([("X", "A"), ("A", "Y"), ("A", "B"), ("Z", "Y")])

    def test_do(self):
        dag_do_x = self.g1.do("A")
        self.assertEqual(set(dag_do_x.nodes()), set(self.g1.nodes()))
        self.assertEqual(sorted(list(dag_do_x.edges())), [("A", "B"), ("A", "Y")])

        dag_do_x = self.g2.do(["A", "Y"])
        self.assertEqual(set(dag_do_x.nodes()), set(self.g2.nodes()))
        self.assertEqual(sorted(list(dag_do_x.edges())), [("A", "B")])


class TestPDAG(unittest.TestCase):
    def setUp(self):
        self.pdag_mix = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
        )
        self.pdag_dir = PDAG(
            directed_ebunch=[("A", "B"), ("D", "B"), ("A", "C"), ("D", "C")]
        )
        self.pdag_undir = PDAG(
            undirected_ebunch=[("A", "C"), ("D", "C"), ("B", "A"), ("B", "D")]
        )
        self.pdag_latent = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["A", "D"],
        )
        self.pdag_role = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            roles={"exposure": "A", "adjustment": "D", "outcome": "C"},
        )
        self.pdag_role_set = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            roles={"exposure": ("A", "D"), "outcome": ("C")},
        )
        self.pdag_role_list = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            roles={"exposure": ["A", "D"], "outcome": ["C"]},
        )

    def test_init_normal(self):
        # Mix directed and undirected
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))

        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(
            directed_ebunch=directed_edges,
            undirected_ebunch=undirected_edges,
            latents=["A", "C"],
        )
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))
        self.assertEqual(pdag.latents, set(["A", "C"]))

        # Only undirected
        undirected_edges = [("A", "C"), ("D", "C"), ("B", "A"), ("B", "D")]
        pdag = PDAG(undirected_ebunch=undirected_edges)
        expected_edges = {
            ("A", "C"),
            ("C", "A"),
            ("D", "C"),
            ("C", "D"),
            ("B", "A"),
            ("A", "B"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set())
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))

        undirected_edges = [("A", "C"), ("D", "C"), ("B", "A"), ("B", "D")]
        pdag = PDAG(undirected_ebunch=undirected_edges, latents=["A", "D"])
        expected_edges = {
            ("A", "C"),
            ("C", "A"),
            ("D", "C"),
            ("C", "D"),
            ("B", "A"),
            ("A", "B"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set())
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))
        self.assertEqual(pdag.latents, set(["A", "D"]))

        # Only directed
        directed_edges = [("A", "B"), ("D", "B"), ("A", "C"), ("D", "C")]
        pdag = PDAG(directed_ebunch=directed_edges)
        self.assertEqual(set(pdag.edges()), set(directed_edges))
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set())

        directed_edges = [("A", "B"), ("D", "B"), ("A", "C"), ("D", "C")]
        pdag = PDAG(directed_ebunch=directed_edges, latents=["D"])
        self.assertEqual(set(pdag.edges()), set(directed_edges))
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set())
        self.assertEqual(pdag.latents, set(["D"]))

    def test_all_neighrors(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.all_neighbors(node="A"), {"B", "C"})
        self.assertEqual(pdag.all_neighbors(node="B"), {"A", "D"})
        self.assertEqual(pdag.all_neighbors(node="C"), {"A", "D"})
        self.assertEqual(pdag.all_neighbors(node="D"), {"B", "C"})

    def test_directed_children(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.directed_children(node="A"), {"C"})
        self.assertEqual(pdag.directed_children(node="B"), set())
        self.assertEqual(pdag.directed_children(node="C"), set())

    def test_directed_parents(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.directed_parents(node="A"), set())
        self.assertEqual(pdag.directed_parents(node="B"), set())
        self.assertEqual(pdag.directed_parents(node="C"), {"A", "D"})

    def test_has_directed_edge(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertTrue(pdag.has_directed_edge("A", "C"))
        self.assertTrue(pdag.has_directed_edge("D", "C"))
        self.assertFalse(pdag.has_directed_edge("A", "B"))
        self.assertFalse(pdag.has_directed_edge("B", "A"))

    def test_has_undirected_edge(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertFalse(pdag.has_undirected_edge("A", "C"))
        self.assertFalse(pdag.has_undirected_edge("D", "C"))
        self.assertTrue(pdag.has_undirected_edge("A", "B"))
        self.assertTrue(pdag.has_undirected_edge("B", "A"))

    def test_undirected_neighbors(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        self.assertEqual(pdag.undirected_neighbors(node="A"), {"B"})
        self.assertEqual(pdag.undirected_neighbors(node="B"), {"A", "D"})
        self.assertEqual(pdag.undirected_neighbors(node="C"), set())
        self.assertEqual(pdag.undirected_neighbors(node="D"), {"B"})

    def test_orient_undirected_edge(self):
        directed_edges = [("A", "C"), ("D", "C")]
        undirected_edges = [("B", "A"), ("B", "D")]
        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

        mod_pdag = pdag.orient_undirected_edge("B", "A", inplace=False)
        self.assertEqual(
            set(mod_pdag.edges()),
            {("A", "C"), ("D", "C"), ("B", "A"), ("B", "D"), ("D", "B")},
        )
        self.assertEqual(mod_pdag.undirected_edges, {("B", "D")})
        self.assertEqual(mod_pdag.directed_edges, {("A", "C"), ("D", "C"), ("B", "A")})

        pdag.orient_undirected_edge("B", "A", inplace=True)
        self.assertEqual(
            set(pdag.edges()),
            {("A", "C"), ("D", "C"), ("B", "A"), ("B", "D"), ("D", "B")},
        )
        self.assertEqual(pdag.undirected_edges, {("B", "D")})
        self.assertEqual(pdag.directed_edges, {("A", "C"), ("D", "C"), ("B", "A")})

        self.assertRaises(
            ValueError, pdag.orient_undirected_edge, "B", "A", inplace=True
        )

    def test_copy(self):
        pdag_copy = self.pdag_mix.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())

        pdag_copy = self.pdag_latent.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set(["A", "D"]))

        pdag_copy = self.pdag_role.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(pdag_copy.get_role("exposure"), ["A"])
        self.assertEqual(pdag_copy.get_role("adjustment"), ["D"])
        self.assertEqual(pdag_copy.get_role("outcome"), ["C"])
        self.assertEqual(
            sorted(pdag_copy.get_roles()), sorted(["adjustment", "exposure", "outcome"])
        )

        pdag_copy = self.pdag_role_set.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(sorted(pdag_copy.get_role("exposure")), sorted(["A", "D"]))
        self.assertEqual(pdag_copy.get_role("outcome"), ["C"])
        self.assertEqual(sorted(pdag_copy.get_roles()), sorted(["exposure", "outcome"]))

        pdag_copy = self.pdag_role_list.copy()
        expected_edges = {
            ("A", "C"),
            ("D", "C"),
            ("A", "B"),
            ("B", "A"),
            ("B", "D"),
            ("D", "B"),
        }
        self.assertEqual(set(pdag_copy.edges()), expected_edges)
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(pdag_copy.directed_edges, set([("A", "C"), ("D", "C")]))
        self.assertEqual(pdag_copy.undirected_edges, set([("B", "A"), ("B", "D")]))
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(sorted(pdag_copy.get_role("exposure")), sorted(["A", "D"]))
        self.assertEqual(pdag_copy.get_role("outcome"), ["C"])
        self.assertEqual(sorted(pdag_copy.get_roles()), sorted(["exposure", "outcome"]))

    def test_pdag_to_dag(self):
        # PDAG no: 1  Possibility of creating a v-structure
        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("C", "D"), ("D", "A")],
        )
        dag = pdag.to_dag()
        self.assertTrue(("A", "B") in dag.edges())
        self.assertTrue(("C", "B") in dag.edges())
        self.assertFalse((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        self.assertTrue(len(dag.edges()) == 4)

        # With latents
        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("C", "D"), ("D", "A")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        self.assertTrue(("A", "B") in dag.edges())
        self.assertTrue(("C", "B") in dag.edges())
        self.assertFalse((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        self.assertEqual(dag.latents, set(["A"]))
        self.assertTrue(len(dag.edges()) == 4)

        # PDAG no: 2  No possibility of creation of v-structure.
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")], undirected_ebunch=[("A", "D")]
        )
        dag = pdag.to_dag()
        self.assertTrue(("B", "C") in dag.edges())
        self.assertTrue(("A", "C") in dag.edges())
        self.assertTrue((("A", "D") in dag.edges()) or (("D", "A") in dag.edges()))

        # With latents
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")],
            undirected_ebunch=[("A", "D")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        self.assertTrue(("B", "C") in dag.edges())
        self.assertTrue(("A", "C") in dag.edges())
        self.assertTrue((("A", "D") in dag.edges()) or (("D", "A") in dag.edges()))
        self.assertEqual(dag.latents, set(["A"]))

        # PDAG no: 3  Already existing v-structure, possibility to add another
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")], undirected_ebunch=[("C", "D")]
        )
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        self.assertEqual(expected_edges, set(dag.edges()))

        # With latents
        pdag = PDAG(
            directed_ebunch=[("B", "C"), ("A", "C")],
            undirected_ebunch=[("C", "D")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        self.assertEqual(expected_edges, set(dag.edges()))
        self.assertEqual(dag.latents, set(["A"]))

        undirected_edges = [(1, 4), (5, 0)]
        directed_edges = [
            (0, 2),
            (1, 2),
            (3, 1),
            (3, 2),
            (3, 4),
            (4, 2),
            (5, 1),
            (5, 2),
            (5, 4),
        ]
        pdag = PDAG(undirected_ebunch=undirected_edges, directed_ebunch=directed_edges)
        dag = pdag.to_dag()
        dag_actual = set(
            [
                (0, 2),
                (1, 2),
                (3, 1),
                (3, 2),
                (3, 4),
                (4, 1),
                (4, 2),
                (5, 0),
                (5, 1),
                (5, 2),
                (5, 4),
            ]
        )
        self.assertSetEqual(set(dag.edges), dag_actual)

    def test_pdag_to_cpdag(self):
        pdag = PDAG(directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C")})

        pdag = PDAG(
            directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C"), ("C", "D")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("C", "D")})

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("D", "C")], undirected_ebunch=[("B", "C")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("D", "C"), ("B", "C"), ("C", "B")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("D", "C"), ("D", "B")],
            undirected_ebunch=[("B", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("D", "C"), ("D", "B"), ("B", "C")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("B", "C")], undirected_ebunch=[("A", "C")]
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C")})

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("B", "C"), ("D", "C")],
            undirected_ebunch=[("A", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C"), ("D", "C")}
        )

        # Examples taken from Perkovi\`c 2017.
        pdag = PDAG(
            directed_ebunch=[("V1", "X")],
            undirected_ebunch=[("X", "V2"), ("V2", "Y"), ("X", "Y")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {("V1", "X"), ("X", "V2"), ("X", "Y"), ("V2", "Y"), ("Y", "V2")},
        )

        pdag = PDAG(
            directed_ebunch=[("Y", "X")],
            undirected_ebunch=[("V1", "X"), ("X", "V2"), ("V2", "Y")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {
                ("X", "V1"),
                ("Y", "X"),
                ("X", "V2"),
                ("V2", "X"),
                ("V2", "Y"),
                ("Y", "V2"),
            },
        )

        # Examples from Bang 2024
        pdag = PDAG(
            directed_ebunch=[("B", "D"), ("C", "D")],
            undirected_ebunch=[("A", "D"), ("A", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True, debug=True)
        self.assertEqual(
            set(cpdag.edges()), {("B", "D"), ("D", "A"), ("C", "A"), ("C", "D")}
        )

        pdag = PDAG(
            directed_ebunch=[("A", "B"), ("C", "B")],
            undirected_ebunch=[("D", "B"), ("D", "A"), ("D", "C")],
        )
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()),
            {
                ("A", "B"),
                ("C", "B"),
                ("D", "B"),
                ("D", "A"),
                ("A", "D"),
                ("D", "C"),
                ("C", "D"),
            },
        )

        undirected_edges = [("A", "C"), ("B", "C"), ("D", "C")]
        directed_edges = [("B", "D"), ("D", "A")]

        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        mpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(mpdag.edges()),
            set(
                [
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

        pdag = PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
        pdag = pdag.apply_meeks_rules()
        self.assertSetEqual(
            set(pdag.edges()),
            set(
                [
                    ("A", "C"),
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

        pdag_inp = PDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )
        pdag_inp.apply_meeks_rules(inplace=True)
        self.assertSetEqual(
            set(pdag_inp.edges()),
            set(
                [
                    ("A", "C"),
                    ("C", "A"),
                    ("C", "B"),
                    ("B", "C"),
                    ("B", "D"),
                    ("D", "A"),
                    ("D", "C"),
                    ("C", "D"),
                ]
            ),
        )

    def test_pdag_equality(self):
        """
        Test the `__eq__` method
        which compares both graph structure and variable-role mappings to allow comparison of two models.
        """
        pdag = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposure": ("A", "D"), "outcome": ["C"]},
        )

        # Case1: When the models are the same
        other1 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposure": ("A", "D"), "outcome": ["C"]},
        )
        # Case2: When the models differ
        other2 = DAG(
            ebunch=[("A", "C"), ("D", "C")],
            latents=["B"],
            roles={"exposure": "A", "adjustment": "D", "outcome": "C"},
        )
        # Case3: When the directed_ebunch variables differ between models
        other3 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C"), ("E", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposure": ("A", "D"), "outcome": ["C"]},
        )
        # Case4: When the directed_ebunch variables differ between models
        other4 = PDAG(
            directed_ebunch=[("A", "E"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposure": ("A", "D"), "outcome": ["C"]},
        )
        # Case5: When the undirected_ebunch variables differ between models
        other5 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "E")],
            latents=["B"],
            roles={"exposure": ("A", "D"), "outcome": ["C"]},
        )
        # Case6: When the latents variables differ between models
        other6 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["D"],
            roles={"exposure": ("A", "D"), "outcome": ["C"]},
        )
        # Case7: When the roles variables differ between models
        other7 = PDAG(
            directed_ebunch=[("A", "C"), ("D", "C")],
            undirected_ebunch=[("B", "A"), ("B", "D")],
            latents=["B"],
            roles={"exposure": ("A"), "adjustment": "D", "outcome": ["C"]},
        )

        self.assertEqual(pdag.__eq__(other1), True)
        self.assertEqual(pdag.__eq__(other2), False)
        self.assertEqual(pdag.__eq__(other3), False)
        self.assertEqual(pdag.__eq__(other4), False)
        self.assertEqual(pdag.__eq__(other5), False)
        self.assertEqual(pdag.__eq__(other6), False)
        self.assertEqual(pdag.__eq__(other7), False)


class TestDAGConversion(unittest.TestCase):
    """Test for DAG to_lavaan and to_dagitty conversion methods"""

    def test_to_lavaan_simple_dag(self):
        """Test conversion of simple DAG to lavaan syntax"""
        dag = DAG([("X", "Y"), ("Z", "Y")])
        result = dag.to_lavaan()
        expected = "Y ~ X + Z"
        self.assertEqual(result, expected)

    def test_to_lavaan_chain_dag(self):
        """Test conversion of chain DAG to lavaan syntax"""
        dag = DAG([("A", "B"), ("B", "C")])
        result = dag.to_lavaan()
        expected = "B ~ A\nC ~ B"
        self.assertEqual(result, expected)

    def test_to_lavaan_complex_dag(self):
        """Test conversion of complex DAG to lavaan syntax"""
        dag = DAG([("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")])
        result = dag.to_lavaan()
        expected = "C ~ A + B\nD ~ A + C"
        self.assertEqual(result, expected)

    def test_to_lavaan_empty_dag(self):
        """Test conversion of empty DAG to lavaan syntax"""
        dag = DAG()
        result = dag.to_lavaan()
        self.assertEqual(result, "")

    def test_to_lavaan_isolated_nodes(self):
        """Test lavaan conversion with isolated nodes (no edges)"""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C"])
        result = dag.to_lavaan()
        self.assertEqual(result, "")  # No edges = no lavaan equations

    def test_to_lavaan_disconnected_components(self):
        """Test lavaan conversion with disconnected components"""
        dag = DAG([("A", "B"), ("C", "D")])
        result = dag.to_lavaan()
        expected = "B ~ A\nD ~ C"
        self.assertEqual(result, expected)

    def test_to_dagitty_simple_dag(self):
        """Test conversion of simple DAG to dagitty syntax"""
        dag = DAG([("X", "Y"), ("Z", "Y")])
        result = dag.to_dagitty()
        expected = "dag {\nX -> Y\nZ -> Y\n}"
        self.assertEqual(result, expected)

    def test_to_dagitty_chain_dag(self):
        """Test conversion of chain DAG to dagitty syntax"""
        dag = DAG([("A", "B"), ("B", "C")])
        result = dag.to_dagitty()
        expected = "dag {\nA -> B\nB -> C\n}"
        self.assertEqual(result, expected)

    def test_to_dagitty_complex_dag(self):
        """Test conversion of complex DAG to dagitty syntax"""
        dag = DAG([("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")])
        result = dag.to_dagitty()
        expected = "dag {\nA -> C\nA -> D\nB -> C\nC -> D\n}"
        self.assertEqual(result, expected)

    def test_to_dagitty_empty_dag(self):
        """Test conversion of empty DAG to dagitty syntax"""
        dag = DAG()
        result = dag.to_dagitty()
        expected = "dag {\n}"
        self.assertEqual(result, expected)

    def test_to_dagitty_isolated_nodes(self):
        """Test dagitty conversion with isolated nodes"""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C"])
        dag.add_edge("A", "B")
        result = dag.to_dagitty()
        expected = "dag {\nA -> B\nC\n}"
        self.assertEqual(result, expected)

    def test_to_dagitty_only_isolated_nodes(self):
        """Test dagitty conversion with only isolated nodes"""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C"])
        result = dag.to_dagitty()
        expected = "dag {\nA\nB\nC\n}"
        self.assertEqual(result, expected)

    def test_to_dagitty_disconnected_components(self):
        """Test dagitty conversion with disconnected components"""
        dag = DAG([("A", "B"), ("C", "D")])
        dag.add_node("E")  # Isolated node
        result = dag.to_dagitty()
        expected = "dag {\nA -> B\nC -> D\nE\n}"
        self.assertEqual(result, expected)

    def test_numeric_node_names(self):
        """Test conversion with numeric node names"""
        dag = DAG([(1, 2), (3, 2)])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "2 ~ 1 + 3"
        self.assertEqual(lavaan_result, expected_lavaan)

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\n1 -> 2\n3 -> 2\n}"
        self.assertEqual(dagitty_result, expected_dagitty)

    def test_mixed_node_name_types(self):
        """Test conversion with mixed node name types"""
        dag = DAG([("A", 1), (2, "B"), (1, 2)])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "1 ~ A\n2 ~ 1\nB ~ 2"
        self.assertEqual(lavaan_result, expected_lavaan)

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\n1 -> 2\n2 -> B\nA -> 1\n}"
        self.assertEqual(dagitty_result, expected_dagitty)

    def test_special_character_node_names(self):
        """Test conversion with special characters in node names"""
        dag = DAG([("node_1", "node_2"), ("var-3", "node_2")])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "node_2 ~ node_1 + var-3"
        self.assertEqual(lavaan_result, expected_lavaan)

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\nnode_1 -> node_2\nvar-3 -> node_2\n}"
        self.assertEqual(dagitty_result, expected_dagitty)

    def test_tuple_node_names(self):
        """Test conversion with tuple node names (hashable objects)"""
        dag = DAG([((1, 2), (3, 4)), ((5, 6), (3, 4))])

        lavaan_result = dag.to_lavaan()
        expected_lavaan = "(3, 4) ~ (1, 2) + (5, 6)"
        self.assertEqual(lavaan_result, expected_lavaan)

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\n(1, 2) -> (3, 4)\n(5, 6) -> (3, 4)\n}"
        self.assertEqual(dagitty_result, expected_dagitty)

    def test_deterministic_output(self):
        """Test that output is deterministic (consistent ordering)"""
        # Create same DAG multiple times with different edge order
        dag1 = DAG([("Z", "Y"), ("X", "Y"), ("A", "B")])
        dag2 = DAG([("A", "B"), ("X", "Y"), ("Z", "Y")])

        self.assertEqual(dag1.to_lavaan(), dag2.to_lavaan())
        self.assertEqual(dag1.to_dagitty(), dag2.to_dagitty())

    def test_single_node_no_edges(self):
        """Test with single node and no edges"""
        dag = DAG()
        dag.add_node("A")

        lavaan_result = dag.to_lavaan()
        self.assertEqual(lavaan_result, "")

        dagitty_result = dag.to_dagitty()
        expected_dagitty = "dag {\nA\n}"
        self.assertEqual(dagitty_result, expected_dagitty)

    def test_self_loops_prevention(self):
        """Test that self-loops are prevented (DAG constraint)"""
        # This should raise an error when creating the DAG
        with self.assertRaisesRegex(ValueError, "Cycles are not allowed"):
            DAG([("A", "A")])

    def test_large_dag(self):
        """Test with a larger DAG to ensure scalability"""
        edges = [(f"X{i}", f"X{i + 1}") for i in range(10)]
        edges.extend([("X0", "X5"), ("X2", "X7")])
        dag = DAG(edges)

        lavaan_result = dag.to_lavaan()
        dagitty_result = dag.to_dagitty()

        # Basic checks
        self.assertIsInstance(lavaan_result, str)
        self.assertIsInstance(dagitty_result, str)
        self.assertGreater(len(lavaan_result), 0)
        self.assertTrue(dagitty_result.startswith("dag {"))
        self.assertTrue(dagitty_result.endswith("}"))

    def test_round_trip_node_preservation(self):
        """Test that all nodes are preserved in some form"""
        # Test with disconnected components and isolated nodes
        dag = DAG([("A", "B"), ("C", "D")])
        dag.add_nodes_from(["E", "F"])  # Isolated nodes

        # For dagitty, all nodes should appear
        dagitty_result = dag.to_dagitty()
        for node in dag.nodes():
            self.assertIn(str(node), dagitty_result)

        # For lavaan, only nodes with edges appear
        lavaan_result = dag.to_lavaan()
        self.assertIn("B", lavaan_result)  # Has parents
        self.assertIn("D", lavaan_result)  # Has parents
        # E and F are isolated, so they won't appear in lavaan

    def test_empty_string_vs_empty_structure(self):
        """Test difference between empty DAG and DAG with no edges"""
        # Completely empty DAG
        empty_dag = DAG()
        self.assertEqual(empty_dag.to_lavaan(), "")
        self.assertEqual(empty_dag.to_dagitty(), "dag {\n}")

        # DAG with nodes but no edges
        nodes_only_dag = DAG()
        nodes_only_dag.add_nodes_from(["A", "B"])
        self.assertEqual(nodes_only_dag.to_lavaan(), "")
        self.assertEqual(nodes_only_dag.to_dagitty(), "dag {\nA\nB\n}")

    def test_complex_multi_parent_structure(self):
        """Test with nodes having multiple parents"""
        dag = DAG(
            [
                ("A", "E"),
                ("B", "E"),
                ("C", "E"),
                ("D", "E"),  # E has 4 parents
                ("E", "F"),
                ("E", "G"),  # E has 2 children
                ("X", "Y"),  # Separate component
            ]
        )

        lavaan_result = dag.to_lavaan()
        self.assertIn("E ~ A + B + C + D", lavaan_result)
        self.assertIn("F ~ E", lavaan_result)
        self.assertIn("G ~ E", lavaan_result)
        self.assertIn("Y ~ X", lavaan_result)

        dagitty_result = dag.to_dagitty()
        self.assertIn("A -> E", dagitty_result)
        self.assertIn("B -> E", dagitty_result)
        self.assertIn("C -> E", dagitty_result)
        self.assertIn("D -> E", dagitty_result)
        self.assertIn("E -> F", dagitty_result)
        self.assertIn("E -> G", dagitty_result)
        self.assertIn("X -> Y", dagitty_result)

    def test_r_compatibility_note(self):
        """Test that node names with spaces/special chars work as documented"""
        # Node names with spaces (should work but may need quoting in R)
        dag_spaces = DAG([("node with spaces", "target")])
        lavaan_result = dag_spaces.to_lavaan()
        dagitty_result = dag_spaces.to_dagitty()

        self.assertEqual("target ~ node with spaces", lavaan_result)
        self.assertEqual("dag {\nnode with spaces -> target\n}", dagitty_result)

    def test_unicode_support(self):
        """Test that unicode characters in node names are supported"""
        dag_unicode = DAG([("α", "β"), ("γ", "β")])
        lavaan_result = dag_unicode.to_lavaan()
        dagitty_result = dag_unicode.to_dagitty()

        self.assertEqual("β ~ α + γ", lavaan_result)
        self.assertEqual("dag {\nα -> β\nγ -> β\n}", dagitty_result)

    def test_complex_dagitty_model(self):
        """Test with complex real-world model from Sebastiani et al. 2005"""
        # Use the exact Sebastiani et al. 2005 model from dagitty.net
        sebastiani_model = """dag {
ADCY9.8 [pos="2.198,9.855"]
ANXA2.11 [pos="0.105,12.599"]
ANXA2.12 [pos="-1.329,13.465"]
ANXA2.13 [pos="1.437,15.415"]
ANXA2.5 [pos="-3.410,11.227"]
ANXA2.7 [pos="-1.418,9.206"]
ANXA2.8 [pos="3.987,20.108"]
BMP6 [pos="-1.985,4.558"]
BMP6.10 [pos="-3.086,2.250"]
BMP6.11 [pos="-1.179,-0.789"]
BMP6.12 [pos="-2.382,-7.072"]
BMP6.13 [pos="-3.207,-4.786"]
BMP6.14 [pos="-4.254,-6.607"]
BMP6.9 [pos="-2.416,-2.349"]
CAT [pos="4.501,12.816"]
CSF2.3 [pos="2.509,-8.195"]
CSF2.4 [pos="-1.283,-8.195"]
ECE1.12 [pos="1.202,8.772"]
ECE1.13 [pos="-0.280,7.260"]
EDN1.10 [pos="-0.282,22.274"]
EDN1.3 [exposure,pos="-2.643,16.642"]
EDN1.9 [pos="-4.064,21.051"]
EDNI1.6 [pos="1.697,21.769"]
EDNI1.7 [outcome,pos="-1.234,18.664"]
MET.5 [pos="4.320,3.489"]
MET.6 [pos="3.562,11.155"]
SELP.12 [pos="3.144,5.234"]
SELP.14 [pos="3.655,-1.126"]
SELP.17 [pos="-1.367,3.939"]
SELP.22 [pos="-0.481,2.644"]
Stroke [pos="1.418,3.285"]
TGFBR3.10 [pos="2.902,-3.702"]
TGFBR3.2 [pos="1.355,-2.058"]
TGFBR3.7 [pos="4.445,-7.336"]
TGFBR3.8 [pos="-0.092,-5.524"]
TGFBR3.9 [pos="1.494,-6.462"]
ANXA2.12 -> ANXA2.11
ANXA2.12 -> ANXA2.13
ANXA2.12 -> ANXA2.5
ANXA2.12 -> ANXA2.7
ANXA2.13 -> ANXA2.11
ANXA2.7 -> ANXA2.11
ANXA2.7 -> ANXA2.5
ANXA2.8 -> ADCY9.8
ANXA2.8 -> ANXA2.13
ANXA2.8 -> BMP6
ANXA2.8 -> CAT
ANXA2.8 -> ECE1.12
ANXA2.8 -> EDN1.3
ANXA2.8 -> EDNI1.6
BMP6 -> BMP6.10
BMP6.10 -> BMP6.11
BMP6.10 -> BMP6.13
BMP6.11 -> BMP6.9
BMP6.13 -> BMP6.12
BMP6.13 -> BMP6.9
BMP6.14 -> BMP6.12
CAT -> MET.5
CSF2.3 -> CSF2.4
ECE1.12 -> ECE1.13
ECE1.13 -> SELP.17
EDN1.10 -> EDNI1.7
EDN1.3 -> ANXA2.12
EDN1.3 -> EDNI1.7
EDN1.9 -> BMP6.14
EDN1.9 -> EDN1.10
EDN1.9 -> EDN1.3
EDNI1.6 -> EDN1.10
EDNI1.6 -> EDNI1.7
MET.5 -> MET.6
MET.5 -> SELP.14
MET.5 -> TGFBR3.7
MET.6 -> SELP.12
SELP.17 -> SELP.22
Stroke -> ADCY9.8
Stroke -> BMP6.10
Stroke -> BMP6.11
Stroke -> BMP6.12
Stroke -> BMP6.13
Stroke -> CSF2.4
Stroke -> ECE1.12
Stroke -> ECE1.13
Stroke -> MET.5
Stroke -> MET.6
Stroke -> SELP.12
Stroke -> SELP.14
Stroke -> SELP.22
Stroke -> TGFBR3.10
Stroke -> TGFBR3.8
TGFBR3.10 -> TGFBR3.2
TGFBR3.10 -> TGFBR3.9
TGFBR3.2 -> TGFBR3.9
TGFBR3.7 -> CSF2.3
TGFBR3.7 -> TGFBR3.10
TGFBR3.8 -> TGFBR3.2
TGFBR3.8 -> TGFBR3.9
}"""

        dag = DAG.from_dagitty(sebastiani_model)

        # Test basic properties
        self.assertEqual(len(dag.nodes()), 36)  # All node variables
        self.assertEqual(
            len(dag.edges()), 60
        )  # Exact number of edges in this complex model
        result_dagitty = dag.to_dagitty()

        self.assertTrue(result_dagitty.startswith("dag {"))
        self.assertTrue(result_dagitty.endswith("}"))

        for node in dag.nodes():
            self.assertIn(str(node), result_dagitty)

        for parent, child in dag.edges():
            edge_str = f"{parent} -> {child}"
            self.assertIn(edge_str, result_dagitty)

        # Convert to lavaan and test structure
        result_lavaan = dag.to_lavaan()

        # Should have many regression equations (one for each non root node)
        lavaan_lines = result_lavaan.strip().split("\n")
        self.assertGreater(len(lavaan_lines), 20)  # many equations for this model

        for line in lavaan_lines:
            self.assertIn(" ~ ", line)
            parts = line.split(" ~ ")
            self.assertEqual(len(parts), 2)
            self.assertGreater(len(parts[0].strip()), 0)  # Nonempty dependent variable
            self.assertGreater(len(parts[1].strip()), 0)  # Nonempty predictors

    def test_round_trip_dagitty_conversion(self):
        """Test round-trip conversion: dagitty -> DAG -> dagitty matches original structure"""
        original_dagitty = """dag {
A [pos="0,0"]
B [pos="1,0"]
C [pos="2,0"]
D [pos="1,1"]
A -> B
B -> C
A -> D
D -> C
}"""

        dag = DAG.from_dagitty(original_dagitty)

        result_dagitty = dag.to_dagitty()
        dag_roundtrip = DAG.from_dagitty(result_dagitty)

        self.assertEqual(set(dag.nodes()), set(dag_roundtrip.nodes()))
        self.assertEqual(set(dag.edges()), set(dag_roundtrip.edges()))

        self.assertIn("A -> B", result_dagitty)
        self.assertIn("B -> C", result_dagitty)
        self.assertIn("A -> D", result_dagitty)
        self.assertIn("D -> C", result_dagitty)

    def test_sebastiani_round_trip_comparison(self):
        """Test that Sebastiani model round-trip preserves graph structure exactly"""
        sebastiani_model = """dag {
ADCY9.8 [pos="2.198,9.855"]
ANXA2.11 [pos="0.105,12.599"]
ANXA2.12 [pos="-1.329,13.465"]
ANXA2.13 [pos="1.437,15.415"]
ANXA2.5 [pos="-3.410,11.227"]
ANXA2.7 [pos="-1.418,9.206"]
ANXA2.8 [pos="3.987,20.108"]
BMP6 [pos="-1.985,4.558"]
BMP6.10 [pos="-3.086,2.250"]
BMP6.11 [pos="-1.179,-0.789"]
BMP6.12 [pos="-2.382,-7.072"]
BMP6.13 [pos="-3.207,-4.786"]
BMP6.14 [pos="-4.254,-6.607"]
BMP6.9 [pos="-2.416,-2.349"]
CAT [pos="4.501,12.816"]
CSF2.3 [pos="2.509,-8.195"]
CSF2.4 [pos="-1.283,-8.195"]
ECE1.12 [pos="1.202,8.772"]
ECE1.13 [pos="-0.280,7.260"]
EDN1.10 [pos="-0.282,22.274"]
EDN1.3 [exposure,pos="-2.643,16.642"]
EDN1.9 [pos="-4.064,21.051"]
EDNI1.6 [pos="1.697,21.769"]
EDNI1.7 [outcome,pos="-1.234,18.664"]
MET.5 [pos="4.320,3.489"]
MET.6 [pos="3.562,11.155"]
SELP.12 [pos="3.144,5.234"]
SELP.14 [pos="3.655,-1.126"]
SELP.17 [pos="-1.367,3.939"]
SELP.22 [pos="-0.481,2.644"]
Stroke [pos="1.418,3.285"]
TGFBR3.10 [pos="2.902,-3.702"]
TGFBR3.2 [pos="1.355,-2.058"]
TGFBR3.7 [pos="4.445,-7.336"]
TGFBR3.8 [pos="-0.092,-5.524"]
TGFBR3.9 [pos="1.494,-6.462"]
ANXA2.12 -> ANXA2.11
ANXA2.12 -> ANXA2.13
ANXA2.12 -> ANXA2.5
ANXA2.12 -> ANXA2.7
ANXA2.13 -> ANXA2.11
ANXA2.7 -> ANXA2.11
ANXA2.7 -> ANXA2.5
ANXA2.8 -> ADCY9.8
ANXA2.8 -> ANXA2.13
ANXA2.8 -> BMP6
ANXA2.8 -> CAT
ANXA2.8 -> ECE1.12
ANXA2.8 -> EDN1.3
ANXA2.8 -> EDNI1.6
BMP6 -> BMP6.10
BMP6.10 -> BMP6.11
BMP6.10 -> BMP6.13
BMP6.11 -> BMP6.9
BMP6.13 -> BMP6.12
BMP6.13 -> BMP6.9
BMP6.14 -> BMP6.12
CAT -> MET.5
CSF2.3 -> CSF2.4
ECE1.12 -> ECE1.13
ECE1.13 -> SELP.17
EDN1.10 -> EDNI1.7
EDN1.3 -> ANXA2.12
EDN1.3 -> EDNI1.7
EDN1.9 -> BMP6.14
EDN1.9 -> EDN1.10
EDN1.9 -> EDN1.3
EDNI1.6 -> EDN1.10
EDNI1.6 -> EDNI1.7
MET.5 -> MET.6
MET.5 -> SELP.14
MET.5 -> TGFBR3.7
MET.6 -> SELP.12
SELP.17 -> SELP.22
Stroke -> ADCY9.8
Stroke -> BMP6.10
Stroke -> BMP6.11
Stroke -> BMP6.12
Stroke -> BMP6.13
Stroke -> CSF2.4
Stroke -> ECE1.12
Stroke -> ECE1.13
Stroke -> MET.5
Stroke -> MET.6
Stroke -> SELP.12
Stroke -> SELP.14
Stroke -> SELP.22
Stroke -> TGFBR3.10
Stroke -> TGFBR3.8
TGFBR3.10 -> TGFBR3.2
TGFBR3.10 -> TGFBR3.9
TGFBR3.2 -> TGFBR3.9
TGFBR3.7 -> CSF2.3
TGFBR3.7 -> TGFBR3.10
TGFBR3.8 -> TGFBR3.2
TGFBR3.8 -> TGFBR3.9
}"""

        original_dag = DAG.from_dagitty(sebastiani_model)

        dagitty_output = original_dag.to_dagitty()
        reconstructed_dag = DAG.from_dagitty(dagitty_output)

        self.assertEqual(set(original_dag.edges()), set(reconstructed_dag.edges()))
        self.assertEqual(set(original_dag.nodes()), set(reconstructed_dag.nodes()))

        for parent, child in original_dag.edges():
            expected_edge_str = f"{parent} -> {child}"
            self.assertIn(
                expected_edge_str, dagitty_output, f"Missing edge: {expected_edge_str}"
            )
