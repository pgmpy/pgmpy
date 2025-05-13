#!/usr/bin/env python3

from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
from pgmpy.base.TimeSeriesPDAG import TimeSeriesPDAG
import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
import networkx as nx


class TestTimeSeriesPDAGCreation(unittest.TestCase):
    def setUp(self):
        self.pdag = TimeSeriesPDAG()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.pdag, TimeSeriesPDAG)
        self.assertEqual(len(self.pdag.nodes()), 0)
        self.assertEqual(len(self.pdag.edges()), 0)

    def test_class_init_with_directed_edges(self):
        directed_edges = [(("A", 0), ("B", 1)), (("B", 1), ("C", 2))]
        self.pdag = TimeSeriesPDAG(directed_ebunch=directed_edges)

        self.assertIsInstance(self.pdag, TimeSeriesPDAG)
        self.assertEqual(len(self.pdag.nodes()), 3)
        self.assertEqual(len(self.pdag.directed_edges), 2)
        self.assertEqual(len(self.pdag.undirected_edges), 0)

        # Check that the directed edges were added correctly
        self.assertListEqual(
            sorted(self.pdag.directed_edges),
            sorted([(("A", 0), ("B", 1)), (("B", 1), ("C", 2))]),
        )

    def test_class_init_with_undirected_edges(self):
        undirected_edges = [(("A", 0), ("C", 0)), (("B", 1), ("D", 1))]
        self.pdag = TimeSeriesPDAG(undirected_ebunch=undirected_edges)

        self.assertIsInstance(self.pdag, TimeSeriesPDAG)
        self.assertEqual(len(self.pdag.nodes()), 4)
        self.assertEqual(len(self.pdag.directed_edges), 0)
        self.assertEqual(len(self.pdag.undirected_edges), 2)

        # Check that the undirected edges were added correctly
        self.assertListEqual(
            sorted(self.pdag.undirected_edges),
            sorted([(("A", 0), ("C", 0)), (("B", 1), ("D", 1))]),
        )

    def test_class_init_with_mixed_edges(self):
        directed_edges = [(("A", 0), ("B", 1)), (("B", 1), ("C", 2))]
        undirected_edges = [(("A", 0), ("C", 0)), (("B", 1), ("D", 1))]
        self.pdag = TimeSeriesPDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )

        self.assertIsInstance(self.pdag, TimeSeriesPDAG)
        self.assertEqual(len(self.pdag.nodes()), 5)
        self.assertEqual(len(self.pdag.directed_edges), 2)
        self.assertEqual(len(self.pdag.undirected_edges), 2)

    def test_class_init_with_latents(self):
        directed_edges = [(("A", 0), ("B", 1))]
        latents = [("A", 0)]
        self.pdag = TimeSeriesPDAG(directed_ebunch=directed_edges, latents=latents)

        self.assertIsInstance(self.pdag, TimeSeriesPDAG)
        self.assertEqual(len(self.pdag.latents), 1)
        self.assertIn(("A", 0), self.pdag.latents)

    def test_validation_node_format(self):
        # Test with incorrect node format
        with self.assertRaises(ValueError):
            TimeSeriesPDAG(directed_ebunch=[("A", ("B", 1))])

        with self.assertRaises(ValueError):
            TimeSeriesPDAG(directed_ebunch=[(("A",), ("B", 1))])

        with self.assertRaises(ValueError):
            TimeSeriesPDAG(directed_ebunch=[(("A", "0"), ("B", 1))])

    def test_validation_temporal_relationship(self):
        # Test with incorrect temporal relationship (later source -> earlier target)
        with self.assertRaises(ValueError):
            TimeSeriesPDAG(directed_ebunch=[(("A", 2), ("B", 1))])

        # Valid: same timepoint
        try:
            TimeSeriesPDAG(directed_ebunch=[(("A", 1), ("B", 1))])
        except ValueError:
            self.fail("TimeSeriesPDAG raised ValueError unexpectedly")

        # Valid: earlier source -> later target
        try:
            TimeSeriesPDAG(directed_ebunch=[(("A", 0), ("B", 1))])
        except ValueError:
            self.fail("TimeSeriesPDAG raised ValueError unexpectedly")

    def test_validation_undirected_edges(self):
        # Test with valid undirected edge at same timepoint
        with self.assertRaises(ValueError):
            TimeSeriesPDAG(undirected_ebunch=[(("A", 0), ("B", 1))])

        # Test with undirected edge between different timepoints (should be valid)
        # try:
        #     TimeSeriesPDAG(undirected_ebunch=[(("A", 0), ("B", 1))])
        # except ValueError:
        #     self.fail("TimeSeriesPDAG raised ValueError unexpectedly")

        # Test with undirected edge that violates temporal constraints
        with self.assertRaises(ValueError):
            TimeSeriesPDAG(undirected_ebunch=[(("A", 2), ("B", 1))])

    def tearDown(self):
        del self.pdag


class TestTimeSeriesPDAGMethods(unittest.TestCase):
    def setUp(self):
        directed_edges = [(("A", 0), ("B", 1)), (("B", 0), ("C", 1))]
        undirected_edges = [(("A", 0), ("C", 0)), (("B", 1), ("D", 1))]
        self.pdag = TimeSeriesPDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )

    def test_get_ancestral_graph(self):
        # Test getting ancestral graph for a node with ancestors
        ancestral = self.pdag.get_ancestral_graph([("B", 1)])

        self.assertIsInstance(ancestral, TimeSeriesPDAG)
        # The ancestral graph should contain 'B' and its ancestors
        self.assertIn(("B", 1), ancestral.nodes())
        self.assertIn(("A", 0), ancestral.nodes())

        # Test getting ancestral graph for a node with no ancestors
        ancestral = self.pdag.get_ancestral_graph([("A", 0)])
        # self.assertEqual(len(ancestral.nodes()), 1)
        self.assertIn(("A", 0), ancestral.nodes())

    def test_get_markov_blanket(self):
        # Test Markov blanket for a node
        mb = self.pdag.get_markov_blanket(("A", 0))

        # 'A' has a directed edge to 'B' and an undirected edge with 'C'
        self.assertIn(("B", 1), mb)
        self.assertIn(("C", 0), mb)

        # Test for node not in graph
        with self.assertRaises(ValueError):
            self.pdag.get_markov_blanket(("Z", 0))

    def test_to_dag(self):
        # Test conversion to TimeSeriesDAG
        ts_dag = self.pdag.to_dag()
        self.assertIsInstance(ts_dag, TimeSeriesDAG)
        self.assertEqual(len(ts_dag.latents), len(self.pdag.latents))

    def test_copy(self):
        pdag_copy = self.pdag.copy()

        self.assertIsInstance(pdag_copy, TimeSeriesPDAG)
        self.assertEqual(len(pdag_copy.nodes()), len(self.pdag.nodes()))
        self.assertEqual(len(pdag_copy.directed_edges), len(self.pdag.directed_edges))
        self.assertEqual(
            len(pdag_copy.undirected_edges), len(self.pdag.undirected_edges)
        )

        # Modifying the copy should not affect the original
        pdag_copy.add_node(("E", 0))
        self.assertNotEqual(len(pdag_copy.nodes()), len(self.pdag.nodes()))

    def test_is_dconnected(self):
        # Simple test case
        self.assertTrue(self.pdag.is_dconnected(("A", 0), ("B", 1)))
        self.assertTrue(self.pdag.is_dconnected(("B", 0), ("C", 1)))

        # Test with observed nodes
        # self.assertTrue(
        #     self.pdag.is_dconnected(("A", 0), ("C", 1), observed=[("B", 0)])
        # )

        # Test with nodes that should be d-separated
        # Path A -> B -> D should be blocked if B is observed
        self.assertFalse(
            self.pdag.is_dconnected(("A", 0), ("D", 1), observed=[("B", 1)])
        )

        # Edge case: test with non-existent nodes
        with self.assertRaises(Exception):
            self.pdag.is_dconnected(("Z", 0), ("C", 0))

        # Test with empty observed set
        self.assertTrue(self.pdag.is_dconnected(("A", 0), ("B", 1), observed=[]))

    def tearDown(self):
        del self.pdag


class TestTimeSeriesPDAGPlotting(unittest.TestCase):
    def setUp(self):
        directed_edges = [
            (("A", 0), ("B", 1)),
            (("B", 1), ("C", 2)),
            (("A", 0), ("D", 1)),
        ]
        undirected_edges = [(("A", 0), ("C", 0)), (("B", 1), ("D", 1))]
        self.pdag = TimeSeriesPDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )

    @patch(
        "matplotlib.pyplot.show"
    )  # Mock plt.show to avoid displaying plots during tests
    def test_plot_summary_graph_basic(self, mock_show):
        summary_graph, fig, ax = self.pdag.plot_summary_graph()

        # Verify the summary graph was created
        self.assertIsInstance(summary_graph, nx.DiGraph)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsNotNone(ax)

        # Check nodes in summary graph
        expected_nodes = {"A", "B", "C", "D"}
        self.assertEqual(set(summary_graph.nodes()), expected_nodes)

        # Check edges and their metadata
        self.assertIn("directed_lags", summary_graph["A"]["B"])
        self.assertIn("undirected_lags", summary_graph["A"]["B"])

        # Test that directed lags are recorded correctly
        self.assertIn((0, 1), summary_graph["A"]["B"]["directed_lags"])

        # Clean up
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_summary_graph_no_undirected(self, mock_show):
        summary_graph, fig, ax = self.pdag.plot_summary_graph(include_undirected=False)

        # Check that undirected edges are not included
        if "A" in summary_graph and "C" in summary_graph:
            if summary_graph.has_edge("A", "C"):
                self.assertFalse(summary_graph["A"]["C"]["undirected_lags"])

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_summary_graph_custom_kwargs(self, mock_show):
        # Test with custom kwargs
        custom_kwargs = {
            "node_kwargs": {"node_color": "blue", "node_size": 800},
            "label_kwargs": {"font_size": 12},
            "directed_edge_kwargs": {"edge_color": "red"},
            "undirected_edge_kwargs": {"edge_color": "green"},
        }

        summary_graph, fig, ax = self.pdag.plot_summary_graph(**custom_kwargs)

        # Not much to assert here without complex figure parsing
        # Just ensure it doesn't crash with custom kwargs
        self.assertIsNotNone(fig)

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_summary_graph_empty_graph(self, mock_show):
        # Test with empty graph
        empty_pdag = TimeSeriesPDAG()
        summary_graph, fig, ax = empty_pdag.plot_summary_graph()

        # Check empty graph properties
        self.assertEqual(len(summary_graph.nodes()), 0)
        self.assertEqual(len(summary_graph.edges()), 0)

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_summary_graph_only_directed(self, mock_show):
        # Test with only directed edges
        directed_pdag = TimeSeriesPDAG(
            directed_ebunch=[(("A", 0), ("B", 1)), (("B", 1), ("C", 2))]
        )
        summary_graph, fig, ax = directed_pdag.plot_summary_graph()

        # Check that all edges have directed lags
        for u, v in summary_graph.edges():
            self.assertTrue(summary_graph[u][v]["directed_lags"])
            self.assertFalse(summary_graph[u][v]["undirected_lags"])

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_summary_graph_only_undirected(self, mock_show):
        # Test with only undirected edges
        undirected_pdag = TimeSeriesPDAG(
            undirected_ebunch=[(("A", 0), ("B", 0)), (("B", 0), ("C", 0))]
        )
        summary_graph, fig, ax = undirected_pdag.plot_summary_graph()

        # Check that all edges have undirected lags
        for u, v in summary_graph.edges():
            self.assertFalse(summary_graph[u][v]["directed_lags"])
            self.assertTrue(summary_graph[u][v]["undirected_lags"])

        plt.close(fig)

    def tearDown(self):
        del self.pdag
