#!/usr/bin/env python3

from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG
from pgmpy.base.TimeSeriesPDAG import TimeSeriesPDAG
import unittest


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
        undirected_edges = [(("A", 0), ("C", 0)), (("B", 0), ("D", 1))]
        self.pdag = TimeSeriesPDAG(undirected_ebunch=undirected_edges)

        self.assertIsInstance(self.pdag, TimeSeriesPDAG)
        self.assertEqual(len(self.pdag.nodes()), 4)
        self.assertEqual(len(self.pdag.directed_edges), 0)
        self.assertEqual(len(self.pdag.undirected_edges), 2)

        # Check that the undirected edges were added correctly
        self.assertListEqual(
            sorted(self.pdag.undirected_edges),
            sorted([(("A", 0), ("C", 0)), (("B", 0), ("D", 1))]),
        )

    def test_class_init_with_mixed_edges(self):
        directed_edges = [(("A", 0), ("B", 1)), (("B", 1), ("C", 2))]
        undirected_edges = [(("A", 0), ("C", 0)), (("B", 0), ("D", 1))]
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
        try:
            TimeSeriesPDAG(undirected_ebunch=[(("A", 0), ("B", 0))])
        except ValueError:
            self.fail("TimeSeriesPDAG raised ValueError unexpectedly")

        # Test with undirected edge between different timepoints (should be valid)
        try:
            TimeSeriesPDAG(undirected_ebunch=[(("A", 0), ("B", 1))])
        except ValueError:
            self.fail("TimeSeriesPDAG raised ValueError unexpectedly")

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

    def test_to_dag(self):
        dag = self.pdag.to_dag()

        self.assertIsInstance(dag, TimeSeriesDAG)
        # The DAG should have the same number of nodes
        self.assertEqual(len(dag.nodes()), len(self.pdag.nodes()))
        # All edges in the DAG should be directed
        self.assertTrue(all(isinstance(edge, tuple) for edge in dag.edges()))

        # The DAG should have at least the same directed edges as the PDAG
        for edge in self.pdag.directed_edges:
            self.assertIn(edge, dag.edges())

    def test_get_ancestral_graph(self):
        # Test getting ancestral graph for a node with ancestors
        ancestral = self.pdag.get_ancestral_graph([("B", 1)])

        self.assertIsInstance(ancestral, TimeSeriesPDAG)
        # The ancestral graph should contain 'B' and its ancestors
        self.assertIn(("B", 1), ancestral.nodes())
        self.assertIn(("A", 0), ancestral.nodes())

        # Test getting ancestral graph for a node with no ancestors
        ancestral = self.pdag.get_ancestral_graph([("A", 0)])
        self.assertEqual(len(ancestral.nodes()), 1)
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

    def test_is_dconnected(self):
        # Test direct connection
        self.assertTrue(self.pdag.is_dconnected(("A", 0), ("C", 0)))

        # Test indirect connection through another node
        self.assertTrue(self.pdag.is_dconnected(("A", 0), ("D", 1)))

        # Test with observed node blocking the path
        self.assertFalse(
            self.pdag.is_dconnected(("A", 0), ("D", 1), observed=[("B", 1)])
        )

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

    def tearDown(self):
        del self.pdag
