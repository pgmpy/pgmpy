import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.estimators import PC
from pgmpy.metrics import (
    SHD,
    self_compatibility_graphical,
)
from pgmpy.metrics.metrics import _latent_admg
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_example_model


class TestLatentADMG(unittest.TestCase):
    """Unit tests for the `_latent_admg` function based on Definition 5 from Faller et al. (2024)."""

    def setUp(self):
        """Prepare canonical DAGs for latent projection tests."""
        # DAG 1: A -> H -> B (H is latent)
        self.dag_chain = DiscreteBayesianNetwork([("A", "H"), ("H", "B")])

        # DAG 2: A -> H <- B (collider through latent H)
        self.dag_collider = DiscreteBayesianNetwork([("A", "H"), ("B", "H")])

    def test_latent_chain_projects_to_directed(self):
        """
        A -> H -> B projects to A -> B when H is latent.
        """
        admg = _latent_admg(self.dag_chain, observed=["A", "B"])
        self.assertEqual(set(admg.edges()), {("A", "B")})

    def test_latent_collider_projects_to_none(self):
        """
        A -> H <- B projects to *no* edge when H is latent.
        """
        admg = _latent_admg(self.dag_collider, observed=["A", "B"])
        # Expect no edge between A and B
        self.assertEqual(set(admg.edges()), set())

    def test_m_graph_unshielded(self):
        """
        M‐structure: a -> x, a -> b <- c, c -> y.  Marginalize to {x,y} => no
        edge between x and y.
        """
        # 1) Build the full M‐structure
        model = DiscreteBayesianNetwork(
            [
                ("a", "x"),
                ("a", "b"),
                ("c", "b"),
                ("c", "y"),
            ]
        )

        # 2) Project onto the observed subset {x,y}
        observed = ["x", "y"]
        admg = _latent_admg(model, observed=observed)

        # 3) Check that only the observed nodes remain
        self.assertEqual(set(admg.nodes()), set(observed))

        # 4) And that no edge has been created between x and y
        self.assertEqual(set(admg.edges()), set())

        # 5) Compare via SHD against an “empty” DAG on {x,y}
        expected = DiscreteBayesianNetwork()  # start with no edges
        expected.add_nodes_from(observed)  # add exactly x,y

        # SHD should be zero when there truly is no edge
        self.assertEqual(SHD(admg, expected), 0)

        # 6) If we now add the spurious edge x -> y, SHD must be non‐zero
        expected.add_edge("x", "y")
        self.assertNotEqual(SHD(admg, expected), 0)

    def test_chain_graph_expected(self):
        """
        Chain structure: a -> b -> c -> d.  Marginalize to {a,c} => a -> c
        """
        # 1) Build the full chain DAG
        model = DiscreteBayesianNetwork(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
            ]
        )

        # 2) Project onto observed subset {a, c}
        observed = ["a", "c"]
        admg = _latent_admg(model, observed=observed)

        # 3) Only nodes a,c should remain
        self.assertEqual(set(admg.nodes()), set(observed))

        # 4) Expect a single directed edge a -> c
        self.assertEqual(set(admg.edges()), {("a", "c")})

        # 5) Compare via SHD against a “true” DBN with exactly that edge
        expected = DiscreteBayesianNetwork()
        expected.add_nodes_from(observed)
        expected.add_edge("a", "c")
        self.assertEqual(SHD(admg, expected), 0)

        # 6) Removing that edge should break equality
        expected.remove_edge("a", "c")
        self.assertNotEqual(SHD(admg, expected), 0)

    def test_unshielded_collider_graph(self):
        """
        Unshielded collider: a -> b <- c -> d.  Marginalize to {a,c} => no edge between a and c
        """
        # 1) Build the collider+extension
        model = DiscreteBayesianNetwork(
            [
                ("a", "b"),
                ("c", "b"),
                ("c", "d"),
            ]
        )

        # 2) Project onto {a,c}
        observed = ["a", "c"]
        admg = _latent_admg(model, observed=observed)

        # 3) Only nodes a,c remain with no connecting edge
        #  self.assertEqual(set(admg.nodes()), set(observed))
        self.assertEqual(set(admg.edges()), set())

        # 4) SHD against empty DBN on {a,c} is zero
        expected = DiscreteBayesianNetwork()
        expected.add_nodes_from(observed)
        self.assertEqual(SHD(admg, expected), 0)

        # 5) Adding a spurious a -> c makes SHD non-zero
        expected.add_edge("a", "c")
        self.assertNotEqual(SHD(admg, expected), 0)

    def test_pure_confounding_graph(self):
        """
        Pure confounding: a -> b and a -> c.  Marginalize to {b,c} => b <-> c
        """
        # 1) Build the confounder DAG
        model = DiscreteBayesianNetwork(
            [
                ("a", "b"),
                ("a", "c"),
            ]
        )

        # 2) Project onto {b,c}
        observed = ["b", "c"]
        admg = _latent_admg(model, observed=observed)

        # 3) Only nodes b,c remain, with a bidirected link b <-> c
        self.assertEqual(set(admg.nodes()), set(observed))
        self.assertEqual(set(admg.edges()), {("b", "c"), ("c", "b")})

        # using nx.DiGraph here only for SHD tests
        expected = nx.DiGraph()
        expected.add_nodes_from(observed)
        expected.add_edge("b", "c")
        expected.add_edge("c", "b")

        # SHD should be zero when they match exactly
        self.assertEqual(SHD(admg, expected), 0)

        # 5) Dropping one direction breaks SHD
        expected.remove_edge("c", "b")
        self.assertNotEqual(SHD(admg, expected), 0)


class TestGraphicalSelfCompatibility(unittest.TestCase):
    """
    Tests for self_compatibility_graphical()
    """

    def setUp(self):
        # A simple A→B→C ground truth
        self.full_dag = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
        rng = np.random.RandomState(0)
        self.data = pd.DataFrame(
            {
                "A": rng.randint(2, size=100),
                "B": rng.randint(2, size=100),
                "C": rng.randint(2, size=100),
            }
        )

    def test_perfect_compatibility(self):
        """
        If joint and all marginals always return the true DAG:
        SHD for each subset = 0 ⇒ mean = 0.0.
        """

        def perfect_factory(df):
            # always returns full_dag
            return SimpleNamespace(estimate=lambda **kw: self.full_dag)

        # Run compatibility
        score = self_compatibility_graphical(
            perfect_factory,
            self.data,
            num_subsets=10,
            subset_fraction=0.5,
            random_state=1,
        )
        # Expect zero distance everywhere
        self.assertEqual(score, 0.0)

    def test_flip_compatibility(self):
        """
        When the joint fit is correct but every marginal is flipped,
        each subset’s SHD should be 2 (two edge reversals), so the mean = 2.0.
        """
        # Capture the ground-truth edges for A→B and B→C
        full_edges = list(self.full_dag.edges())  # [('A','B'), ('B','C')]
        state = {"calls": 0}

        def flip_factory(df):
            cols = list(df.columns)  # e.g. ["A","B","C"]

            def estimate(**kw):
                # Track how many times estimate() is called
                state["calls"] += 1

                # Build a DAG on the same node set
                dag = DiscreteBayesianNetwork([])
                # Add all nodes
                for node in cols:
                    dag.add_node(node)

                # First call → joint → correct orientation
                if state["calls"] == 1:
                    for u, v in full_edges:
                        dag.add_edge(u, v)
                else:
                    # Subsequent calls → marginals → flipped orientation
                    for u, v in full_edges:
                        dag.add_edge(v, u)

                return dag

            return SimpleNamespace(estimate=estimate)

        # Compute graphical compatibility over full-node subsets
        score = self_compatibility_graphical(
            flip_factory,
            self.data,
            num_subsets=10,
            subset_fraction=1.0,  # use all nodes each time
            random_state=0,
        )

        # Now every marginal is the reverse of the joint → SHD = 2 per subset
        self.assertAlmostEqual(score, 2.0, places=6)

    def test_kwargs_forwarded(self):
        """
        Ensure that arbitrary kwargs (e.g. alpha, foo) are passed through
        exactly once to .estimate() when no marginal runs occur (num_subsets=0).
        """
        seen = {}

        def record_factory(df):
            def estimate(**kw):
                seen.update(kw)
                # We can return anything since no subsets will be processed
                return DiscreteBayesianNetwork([])

            return SimpleNamespace(estimate=estimate)

        # Use num_subsets=0 so we only invoke the joint estimator once
        _ = self_compatibility_graphical(
            record_factory,
            self.data,
            num_subsets=0,
            subset_fraction=1.0,
            random_state=3,
            alpha=0.01,
            foo="bar",
        )
        self.assertIn("alpha", seen)
        self.assertEqual(seen["alpha"], 0.01)
        self.assertIn("foo", seen)
        self.assertEqual(seen["foo"], "bar")

    def test_perfect_subset_projection(self):
        """
        If both joint and marginal estimators always return the exact
        Definition-5 projection of the true model onto S, then SHD=0.
        """
        # 1) True “full” model A→B→C→D
        true_model = DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("C", "D")])

        # 2) Factory that returns the *latent-projected* subgraph on df.columns
        def proj_factory(df):
            def estimate(**kw):
                S = set(df.columns)
                # Leverage your _latent_admg implementation on the true model
                g = _latent_admg(true_model, list(S))
                # Convert back to a pgmpy BayesianModel for SHD()
                m = DiscreteBayesianNetwork([])
                m.add_nodes_from(g.nodes())
                for u, v in g.edges():
                    m.add_edge(u, v)
                return m

            return SimpleNamespace(estimate=estimate)

        # 3) Dummy data (values irrelevant)
        dummy = pd.DataFrame(
            {
                "A": np.zeros(50),
                "B": np.zeros(50),
                "C": np.zeros(50),
                "D": np.zeros(50),
            }
        )

        # 4) Compute compatibility over proper subsets
        score = self_compatibility_graphical(
            proj_factory,
            dummy,
            num_subsets=20,
            subset_fraction=0.75,
            random_state=42,
        )

        # Now joint_proj == marg_proj for every draw ⇒ mean SHD = 0
        self.assertEqual(score, 0.0)

    def test_child_example_low_score(self):
        """
        When fitting the small Child network on its own simulated data,
        the self-compatibility score should be very low.
        """
        # 1) Load the small Child example and simulate
        model = get_example_model("child")
        data = model.simulate(n_samples=50, seed=42)

        # 2) Compute graphical self-compatibility
        score = self_compatibility_graphical(
            PC,
            data,
            num_subsets=20,
            subset_fraction=0.8,
            random_state=42,
            scoring_method="bic-d",
        )
        # expect an shd below 5.0 but not 0.0
        self.assertGreater(score, 0.0, "SHD should be positive")
        self.assertLess(score, 5.0, "SHD should be less than 5")
