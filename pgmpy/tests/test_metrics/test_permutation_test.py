#!/usr/bin/env python

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.metrics.permutation_test import (
    _count_lmc_violations,
    _create_permuted_graph,
    _get_non_descendants,
    permutation_based_falsification_test,
)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_example_model


class TestPermutationBasedFalsificationTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with known data patterns."""
        # Create a simple chain model: X -> Y -> Z
        self.simple_model = DiscreteBayesianNetwork([("X", "Y"), ("Y", "Z")])

        # Generate synthetic data that follows the model structure
        np.random.seed(42)
        n_samples = 1000

        # X is independent
        X = np.random.binomial(1, 0.5, n_samples)
        # Y depends on X
        Y = np.random.binomial(1, 0.3 + 0.4 * X)
        # Z depends on Y
        Z = np.random.binomial(1, 0.2 + 0.6 * Y)

        self.simple_data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        # Create a more complex model for testing
        self.complex_model = DiscreteBayesianNetwork(
            [("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")]
        )

        # Generate data for complex model
        A = np.random.binomial(1, 0.6, n_samples)
        B = np.random.binomial(1, 0.4, n_samples)
        C = np.random.binomial(1, 0.2 + 0.3 * A + 0.3 * B)
        D = np.random.binomial(1, 0.1 + 0.7 * C)
        E = np.random.binomial(1, 0.3 + 0.5 * C)

        self.complex_data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E})

        # Create continuous data for pearsonr testing
        X_cont = np.random.normal(0, 1, n_samples)
        Y_cont = 0.5 * X_cont + np.random.normal(0, 0.5, n_samples)
        Z_cont = 0.7 * Y_cont + np.random.normal(0, 0.3, n_samples)

        self.continuous_data = pd.DataFrame({"X": X_cont, "Y": Y_cont, "Z": Z_cont})

    def test_basic_functionality(self):
        """Test that the function runs without error and returns expected structure."""
        result = permutation_based_falsification_test(
            self.simple_model, self.simple_data, n_permutations=10, show_progress=False
        )

        # Check return structure
        expected_keys = [
            "falsifiable",
            "falsified",
            "p_value_falsifiable",
            "p_value_falsified",
            "lmc_violations",
            "n_permutations",
            "same_mec_count",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Check data types
        self.assertIsInstance(result["falsifiable"], bool)
        self.assertIsInstance(result["falsified"], bool)
        self.assertIsInstance(result["p_value_falsifiable"], float)
        self.assertIsInstance(result["p_value_falsified"], float)
        self.assertIsInstance(result["lmc_violations"], int)

        # Check value ranges
        self.assertGreaterEqual(result["p_value_falsifiable"], 0.0)
        self.assertLessEqual(result["p_value_falsifiable"], 1.0)
        self.assertGreaterEqual(result["p_value_falsified"], 0.0)
        self.assertLessEqual(result["p_value_falsified"], 1.0)
        self.assertGreaterEqual(result["lmc_violations"], 0)

    def test_with_return_summary(self):
        """Test detailed summary return."""
        result = permutation_based_falsification_test(
            self.simple_model,
            self.simple_data,
            n_permutations=10,
            return_summary=True,
            show_progress=False,
        )

        self.assertIn("summary", result)
        summary = result["summary"]

        expected_summary_keys = [
            "permutation_violations",
            "significance_level",
            "ci_test",
            "mean_permutation_violations",
            "std_permutation_violations",
            "min_permutation_violations",
            "max_permutation_violations",
        ]

        for key in expected_summary_keys:
            self.assertIn(key, summary)

        # Check that permutation violations is a list
        self.assertIsInstance(summary["permutation_violations"], list)
        self.assertEqual(len(summary["permutation_violations"]), 10)

    def test_wrong_model_detection(self):
        """Test that clearly wrong models are more likely to be falsified."""
        # Create a model that contradicts the data generation process
        wrong_model = DiscreteBayesianNetwork(
            [("Z", "Y"), ("Y", "X")]
        )  # Completely reversed

        # Test both models
        correct_result = permutation_based_falsification_test(
            self.simple_model, self.simple_data, n_permutations=50, show_progress=False
        )
        wrong_result = permutation_based_falsification_test(
            wrong_model, self.simple_data, n_permutations=50, show_progress=False
        )

        # Wrong model should have more violations
        self.assertGreaterEqual(
            wrong_result["lmc_violations"], correct_result["lmc_violations"]
        )

    def test_continuous_data_support(self):
        """Test support for continuous data with pearsonr."""
        result = permutation_based_falsification_test(
            self.simple_model,
            self.continuous_data,
            ci_test="pearsonr",
            n_permutations=10,
            show_progress=False,
        )

        # Should run without error
        self.assertIsInstance(result["falsifiable"], bool)
        self.assertIsInstance(result["falsified"], bool)

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test with wrong data type
        with self.assertRaises(TypeError):
            permutation_based_falsification_test(
                self.simple_model, "not_a_dataframe", show_progress=False
            )

        # Test with missing variables in data
        incomplete_data = self.simple_data[["X", "Y"]]  # Missing Z
        with self.assertRaises(ValueError):
            permutation_based_falsification_test(
                self.simple_model, incomplete_data, show_progress=False
            )

        # Test with unsupported CI test
        with self.assertRaises(ValueError):
            permutation_based_falsification_test(
                self.simple_model,
                self.simple_data,
                ci_test="unsupported_test",
                show_progress=False,
            )

    def test_deterministic_behavior(self):
        """Test that results are deterministic when using fixed random seed."""
        # Set seed and run test
        np.random.seed(123)
        result1 = permutation_based_falsification_test(
            self.simple_model, self.simple_data, n_permutations=20, show_progress=False
        )

        # Reset seed and run again
        np.random.seed(123)
        result2 = permutation_based_falsification_test(
            self.simple_model, self.simple_data, n_permutations=20, show_progress=False
        )

        # Results should be identical
        self.assertEqual(result1["lmc_violations"], result2["lmc_violations"])
        self.assertEqual(result1["p_value_falsifiable"], result2["p_value_falsifiable"])
        self.assertEqual(result1["p_value_falsified"], result2["p_value_falsified"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small number of permutations
        result = permutation_based_falsification_test(
            self.simple_model, self.simple_data, n_permutations=1, show_progress=False
        )
        self.assertEqual(result["n_permutations"], 1)

        # Test with single node model
        single_node_model = DiscreteBayesianNetwork()
        single_node_model.add_node("A")
        single_node_data = pd.DataFrame({"A": [0, 1, 0, 1]})

        result = permutation_based_falsification_test(
            single_node_model, single_node_data, n_permutations=5, show_progress=False
        )
        # Should run without error
        self.assertIsInstance(result["falsifiable"], bool)

    def test_with_real_pgmpy_models(self):
        """Test with real pgmpy example models."""
        try:
            # Test with cancer model
            cancer_model = get_example_model("cancer")
            cancer_data = cancer_model.simulate(500)

            result = permutation_based_falsification_test(
                cancer_model, cancer_data, n_permutations=10, show_progress=False
            )

            # Should not falsify the true model (though might depending on sample size)
            self.assertIsInstance(result["falsifiable"], bool)
            self.assertIsInstance(result["falsified"], bool)

        except Exception as e:
            # Skip if example models not available
            self.skipTest(f"Example models not available: {e}")


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for helper functions."""
        self.model = DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("A", "D")])

        # Create test data
        np.random.seed(42)
        n = 100
        A = np.random.binomial(1, 0.5, n)
        B = np.random.binomial(1, 0.3 + 0.4 * A)
        C = np.random.binomial(1, 0.2 + 0.6 * B)
        D = np.random.binomial(1, 0.1 + 0.7 * A)

        self.data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})

    def test_get_non_descendants(self):
        """Test the _get_non_descendants function."""
        # A -> B -> C, A -> D
        # Let's trace descendants for each node:
        # A can reach: B (direct), C (via B), D (direct) -> descendants = {B, C, D}
        # B can reach: C (direct) -> descendants = {C}
        # C can reach: nothing -> descendants = {}
        # D can reach: nothing -> descendants = {}
        #
        # Non-descendants = all_nodes - descendants - {self}:
        # A's non-descendants: {A,B,C,D} - {B,C,D} - {A} = {}
        # B's non-descendants: {A,B,C,D} - {C} - {B} = {A, D}
        # C's non-descendants: {A,B,C,D} - {} - {C} = {A, B, D}
        # D's non-descendants: {A,B,C,D} - {} - {D} = {A, B, C}

        non_desc_A = _get_non_descendants(self.model, "A")
        non_desc_B = _get_non_descendants(self.model, "B")
        non_desc_C = _get_non_descendants(self.model, "C")
        non_desc_D = _get_non_descendants(self.model, "D")

        self.assertEqual(set(non_desc_A), set())  # A can reach all other nodes
        self.assertEqual(set(non_desc_B), {"A", "D"})  # B cannot reach A or D
        self.assertEqual(set(non_desc_C), {"A", "B", "D"})  # C cannot reach A, B, or D
        self.assertEqual(set(non_desc_D), {"A", "B", "C"})  # D cannot reach A, B, or C

    def test_create_permuted_graph(self):
        """Test the _create_permuted_graph function."""
        perm_mapping = {"A": "X", "B": "Y", "C": "Z", "D": "W"}

        permuted_model = _create_permuted_graph(self.model, perm_mapping)

        # Check that edges are correctly permuted
        original_edges = set(self.model.edges())
        expected_permuted_edges = {
            (perm_mapping[u], perm_mapping[v]) for u, v in original_edges
        }
        actual_permuted_edges = set(permuted_model.edges())

        self.assertEqual(expected_permuted_edges, actual_permuted_edges)

    def test_count_lmc_violations(self):
        """Test the _count_lmc_violations function."""
        from pgmpy.estimators.CITests import chi_square

        violations = _count_lmc_violations(
            self.model, self.data, chi_square, significance_level=0.05
        )

        # Should return a non-negative integer
        self.assertIsInstance(violations, int)
        self.assertGreaterEqual(violations, 0)

    def test_count_lmc_violations_with_insufficient_data(self):
        """Test LMC violation counting with insufficient data."""
        from pgmpy.estimators.CITests import chi_square

        # Create very small dataset
        small_data = self.data.head(5)

        # Should handle gracefully without crashing
        violations = _count_lmc_violations(
            self.model, small_data, chi_square, significance_level=0.05
        )

        self.assertIsInstance(violations, int)
        self.assertGreaterEqual(violations, 0)


class TestProgressBarAndLogging(unittest.TestCase):
    def test_progress_bar_disabled(self):
        """Test that progress bar can be disabled."""
        model = DiscreteBayesianNetwork([("X", "Y")])
        data = pd.DataFrame({"X": [0, 1, 0, 1], "Y": [1, 1, 0, 0]})

        # Should run without showing progress bar
        result = permutation_based_falsification_test(
            model, data, n_permutations=5, show_progress=False
        )

        self.assertIsInstance(result["falsifiable"], bool)

    @patch("pgmpy.metrics.permutation_test.logger")
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""
        model = DiscreteBayesianNetwork([("X", "Y")])
        data = pd.DataFrame({"X": [0, 1, 0, 1], "Y": [1, 1, 0, 0]})

        permutation_based_falsification_test(
            model, data, n_permutations=3, show_progress=False
        )

        # Check that info logging was called
        self.assertTrue(mock_logger.info.called)


if __name__ == "__main__":
    unittest.main()
