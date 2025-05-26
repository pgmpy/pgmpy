import unittest

import numpy as np
import pandas as pd

from pgmpy.base import IPDAG
from pgmpy.estimators import GIES


class TestGIES(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with interventions
        np.random.seed(42)
        n_samples = 1000

        # Generate data with a known structure: A -> B -> C
        data = pd.DataFrame(np.random.randn(n_samples, 3), columns=["A", "B", "C"])
        data["B"] = 0.5 * data["A"] + 0.1 * np.random.randn(n_samples)
        data["C"] = 0.7 * data["B"] + 0.1 * np.random.randn(n_samples)

        # Add intervention targets
        data["intervention_targets"] = [None] * 800 + ["A"] * 100 + ["B"] * 100

        self.data = data
        self.gies = GIES(data)

    def test_init(self):
        """Test initialization of GIES estimator."""
        self.assertIsInstance(self.gies, GIES)
        self.assertEqual(self.gies.scoring_method, "bic-g")

        # Test error for invalid data type
        with self.assertRaises(TypeError):
            GIES(np.random.randn(100, 3))

        # Test error for missing intervention_targets column
        with self.assertRaises(ValueError):
            GIES(pd.DataFrame(np.random.randn(100, 3)))

    def test_estimate(self):
        """Test structure estimation."""
        # Estimate the structure
        ipdag = self.gies.estimate()

        # Check that we get an IPDAG
        self.assertIsInstance(ipdag, IPDAG)

        # Check that the intervention targets are preserved
        self.assertEqual(set(ipdag.intervention_targets), {None, "A", "B"})

        # Check that all variables are in the graph
        self.assertEqual(set(ipdag.nodes()), {"A", "B", "C"})

        # Check that we can get a DAG
        dag = self.gies.estimate(return_type="dag")
        self.assertIsNotNone(dag)

        # Test invalid return_type
        with self.assertRaises(ValueError):
            self.gies.estimate(return_type="invalid")

    def test_score_operation(self):
        """Test scoring of operations."""
        # Create a simple IPDAG
        ipdag = IPDAG(
            directed_ebunch=[("A", "B")],
            undirected_ebunch=[("B", "C")],
            intervention_targets=[None, "A", "B"],
        )

        # Test scoring different operations
        add_score = self.gies._score_operation(ipdag, "add", "A", "C")
        remove_score = self.gies._score_operation(ipdag, "remove", "A", "B")
        reverse_score = self.gies._score_operation(ipdag, "reverse", "A", "B")

        self.assertIsInstance(add_score, float)
        self.assertIsInstance(remove_score, float)
        self.assertIsInstance(reverse_score, float)

    def test_interventional_constraints(self):
        """Test that the algorithm respects interventional constraints."""
        # Create data with a known structure and interventions
        data = pd.DataFrame(np.random.randn(1000, 3), columns=["A", "B", "C"])
        data["intervention_targets"] = [None] * 800 + ["A"] * 200

        gies = GIES(data)
        ipdag = gies.estimate()

        # Check that A is not a child of any other variable (due to interventions)
        for node in ["B", "C"]:
            self.assertFalse(ipdag.has_edge(node, "A"))


if __name__ == "__main__":
    unittest.main()
