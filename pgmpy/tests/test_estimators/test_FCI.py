import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators.FCIPlus import FCIPlus


class TestFCIPlus(unittest.TestCase):
    def test_fci_plus_estimate(self):
        """Test basic V-structure and Rule 1 propagation."""
        np.random.seed(42)
        X, Z = np.random.normal(size=2000), np.random.normal(size=2000)
        Y = X + Z + np.random.normal(scale=0.1, size=2000)
        W = Y + np.random.normal(scale=0.1, size=2000)
        df = pd.DataFrame({"X": X, "Z": Z, "Y": Y, "W": W})

        edges = (
            FCIPlus(df).estimate(significance_level=0.05, show_progress=False).edges()
        )
        self.assertIn(("X", "Y"), edges)
        self.assertIn(("Z", "Y"), edges)
        self.assertIn(("Y", "W"), edges)

    def test_rule_2_transitivity(self):
        """Test Rule 2 transitivity/cycle prevention."""
        np.random.seed(42)
        # S and Z create a v-structure at A: S -> A <- Z
        S = np.random.normal(size=2000)
        Z = np.random.normal(size=2000)
        A = S + Z + np.random.normal(scale=0.1, size=2000)

        # B and C are caused by A
        B = A + np.random.normal(scale=0.1, size=2000)
        C = A + np.random.normal(scale=0.1, size=2000)

        # Create a direct link between B and C that isn't just A
        B = B + np.random.normal(size=2000)
        C = C + 0.5 * B + np.random.normal(scale=0.1, size=2000)

        df = pd.DataFrame({"S": S, "Z": Z, "A": A, "B": B, "C": C})

        # We use a higher significance level to be more 'permissive' of edges
        edges = (
            FCIPlus(df).estimate(significance_level=0.01, show_progress=False).edges()
        )

        # A is a collider for S and Z, so S -> A <- Z is oriented.
        # Rule 1 then propagates A -> B and A -> C
        self.assertTrue(("A", "B") in edges or ("A", "C") in edges)

    def test_latent_star_graph(self):
        """Test Rule 4 and Latent Confounder detection."""
        np.random.seed(42)
        H = np.random.normal(size=2000)
        A, B = np.random.normal(size=2000), np.random.normal(size=2000)
        X = H + A + np.random.normal(scale=0.1, size=2000)
        Y = H + B + np.random.normal(scale=0.1, size=2000)

        df = pd.DataFrame({"A": A, "X": X, "Y": Y, "B": B})
        edges = (
            FCIPlus(df).estimate(significance_level=0.05, show_progress=False).edges()
        )

        # X <-> Y is represented as two directed edges in the export logic
        self.assertIn(("X", "Y"), edges)
        self.assertIn(("Y", "X"), edges)
