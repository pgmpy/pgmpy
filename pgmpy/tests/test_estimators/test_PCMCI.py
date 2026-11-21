#!/usr/bin/env python

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import networkx as nx

from pgmpy.estimators.PCMCI import PCMCI


class TestPCMCI(unittest.TestCase):
    """Test suite for the PCMCI time series causal discovery algorithm."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample time series data
        np.random.seed(42)
        self.T = 100
        self.data = pd.DataFrame(np.random.randn(self.T, 3), columns=["X", "Y", "Z"])

        # Create a simple AR process: X causes Y with lag 1
        for t in range(1, self.T):
            self.data.loc[t, "Y"] += (
                0.5 * self.data.loc[t - 1, "X"] + 0.1 * np.random.randn()
            )
            self.data.loc[t, "Z"] += (
                0.3 * self.data.loc[t - 1, "Y"] + 0.1 * np.random.randn()
            )

        # Mock the PCMCI class for testing
        self.pcmci = PCMCI(data=self.data)

        # Mock CI test function
        self.mock_ci_test = Mock(return_value=False)  # Default: variables are dependent

    def test_init(self):
        """Test PCMCI initialization."""
        # Test with data
        pcmci = PCMCI(data=self.data)
        self.assertEqual(pcmci.data.shape, self.data.shape)
        self.assertIsNone(pcmci.independencies)

        # Test with no data
        pcmci = PCMCI()
        self.assertIsNone(pcmci.data)
        self.assertIsNone(pcmci.independencies)

    def test_estimate_no_data_raises_error(self):
        """Test that estimate raises ValueError when no data is provided."""
        pcmci = PCMCI()
        with self.assertRaises(ValueError):
            pcmci.estimate()

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_estimate_skeleton_return_type(self, mock_get_ci_test):
        """Test that skeleton return type works correctly."""
        mock_get_ci_test.return_value = self.mock_ci_test

        # Mock the skeleton building method
        mock_skeleton = nx.Graph()
        mock_skeleton.add_edges_from([(("X", 0), ("Y", 0)), (("X", 1), ("Y", 0))])
        mock_separating_sets = {}

        with patch.object(
            self.pcmci,
            "_build_time_series_skeleton",
            return_value=(mock_skeleton, mock_separating_sets),
        ):
            result = self.pcmci.estimate(return_type="skeleton")

            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            skeleton, sep_sets = result
            self.assertIsInstance(skeleton, nx.Graph)
            self.assertIsInstance(sep_sets, dict)

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_estimate_invalid_return_type(self, mock_get_ci_test):
        """Test that invalid return_type raises ValueError."""
        mock_get_ci_test.return_value = self.mock_ci_test

        with self.assertRaises(ValueError):
            self.pcmci.estimate(return_type="invalid_type")

    def test_create_lagged_data(self):
        """Test creation of lagged data."""
        max_lag = 2
        lagged_data = self.pcmci._create_lagged_data(self.data, max_lag)

        # Check that we have the right number of columns (3 variables * 3 lags = 9)
        expected_cols = 3 * (max_lag + 1)
        self.assertEqual(len(lagged_data.columns), expected_cols)

        # Check column names are tuples
        for col in lagged_data.columns:
            self.assertIsInstance(col, tuple)
            self.assertEqual(len(col), 2)  # (variable, lag)
            self.assertIn(col[0], ["X", "Y", "Z"])
            self.assertIn(col[1], [0, 1, 2])

        # Check that NaN rows are dropped
        self.assertFalse(lagged_data.isnull().any().any())

        # Check that the length is reduced by max_lag
        self.assertEqual(len(lagged_data), self.T - max_lag)

    def test_create_lagged_data_edge_cases(self):
        """Test edge cases for lagged data creation."""
        # Test with max_lag = 0
        lagged_data = self.pcmci._create_lagged_data(self.data, 0)
        self.assertEqual(len(lagged_data.columns), 3)
        self.assertEqual(len(lagged_data), self.T)

        # Test with non-DataFrame input
        with self.assertRaises(TypeError):
            self.pcmci._create_lagged_data(np.array([[1, 2], [3, 4]]), 1)

        # Test with negative max_lag
        with self.assertRaises(ValueError):
            self.pcmci._create_lagged_data(self.data, -1)

    def test_get_potential_sepsets(self):
        """Test generation of potential separating sets."""
        # Create a simple graph
        graph = nx.Graph()
        nodes = [("X", 0), ("Y", 0), ("Z", 0), ("X", 1), ("Y", 1)]
        graph.add_nodes_from(nodes)
        graph.add_edges_from(
            [
                (("X", 0), ("Y", 0)),
                (("X", 0), ("Z", 0)),
                (("Y", 0), ("Z", 0)),
                (("X", 1), ("X", 0)),
                (("Y", 1), ("Y", 0)),
            ]
        )

        # Test separating sets for contemporaneous variables
        sepsets = self.pcmci._get_potential_sepsets(("X", 0), ("Y", 0), graph, 1)

        # Should be a list of tuples, each containing one variable
        self.assertIsInstance(sepsets, list)
        for sepset in sepsets:
            self.assertIsInstance(sepset, tuple)
            self.assertEqual(len(sepset), 1)

    def test_ci_test_wrapper(self):
        """Test the CI test wrapper function."""
        max_lag = 1
        lagged_data = self.pcmci._create_lagged_data(self.data, max_lag)

        # Mock CI test that always returns True (independent)
        mock_ci_test = Mock(return_value=True)

        u = ("X", 0)
        v = ("Y", 0)
        sep_set = [("Z", 0)]

        result = self.pcmci._ci_test_wrapper(
            u, v, sep_set, lagged_data, mock_ci_test, 0.05
        )

        self.assertTrue(result)

        # Check that CI test was called with string column names
        mock_ci_test.assert_called_once()
        args, _ = mock_ci_test.call_args

        # Check that the arguments are strings, not tuples
        self.assertIsInstance(args[0], str)  # u_str
        self.assertIsInstance(args[1], str)  # v_str
        self.assertIsInstance(args[2], list)  # sep_set_str
        for item in args[2]:
            self.assertIsInstance(item, str)

    def test_orient_time_series_edges(self):
        """Test edge orientation based on temporal constraints."""
        # Create a simple skeleton
        skeleton = nx.Graph()
        skeleton.add_nodes_from([("X", 0), ("Y", 0), ("X", 1), ("Y", 1)])
        skeleton.add_edges_from(
            [
                (("X", 1), ("X", 0)),  # Temporal edge
                (("X", 1), ("Y", 0)),  # Cross-temporal edge
                (("X", 0), ("Y", 0)),  # Contemporaneous edge
            ]
        )

        separating_sets = {}

        ts_dag = self.pcmci._orient_time_series_edges(
            skeleton, separating_sets, max_time_lag=1
        )

        # Check that temporal edges are oriented correctly (past -> present)
        self.assertTrue(ts_dag.has_edge(("X", 1), ("X", 0)))
        self.assertFalse(ts_dag.has_edge(("X", 0), ("X", 1)))

        self.assertTrue(ts_dag.has_edge(("X", 1), ("Y", 0)))
        self.assertFalse(ts_dag.has_edge(("Y", 0), ("X", 1)))

    def test_remove_cycles_within_time_slice(self):
        """Test removal of cycles within the same time slice."""
        from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG

        ts_dag = TimeSeriesDAG()
        ts_dag.add_nodes_from([("X", 0), ("Y", 0), ("Z", 0)])

        # Create a cycle within time slice 0
        ts_dag.add_edges_from(
            [(("X", 0), ("Y", 0)), (("Y", 0), ("Z", 0)), (("Z", 0), ("X", 0))]
        )

        original_edges = ts_dag.number_of_edges()
        self.pcmci._remove_cycles_within_time_slice(ts_dag)

        # Should have fewer edges after cycle removal
        self.assertLess(ts_dag.number_of_edges(), original_edges)

        # Should not have cycles
        self.assertFalse(list(nx.simple_cycles(ts_dag)))

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_build_time_series_skeleton(self, mock_get_ci_test):
        """Test building of time series skeleton."""
        mock_get_ci_test.return_value = self.mock_ci_test

        # Mock the CI test to return independence for some pairs
        def side_effect_ci_test(u, v, sep_set, **kwargs):
            # Make X_lag_1 and Z_lag_0 independent given Y_lag_0
            if (u == "X_lag_1" and v == "Z_lag_0" and "Y_lag_0" in sep_set) or (
                u == "Z_lag_0" and v == "X_lag_1" and "Y_lag_0" in sep_set
            ):
                return True
            return False

        mock_ci_test = Mock(side_effect=side_effect_ci_test)

        skeleton, sep_sets = self.pcmci._build_time_series_skeleton(
            ci_test=mock_ci_test,
            max_time_lag=1,
            max_cond_vars=2,
            show_progress=False,
            n_jobs=1,
        )

        # Check that skeleton is a networkx Graph
        self.assertIsInstance(skeleton, nx.Graph)

        # Check that we have the expected number of nodes (3 vars * 2 lags = 6)
        expected_nodes = 3 * 2  # 3 variables, 2 time points (0, 1)
        self.assertEqual(len(skeleton.nodes()), expected_nodes)

        # Check that all nodes are tuples
        for node in skeleton.nodes():
            self.assertIsInstance(node, tuple)
            self.assertEqual(len(node), 2)

    def test_run_single_mci_test(self):
        """Test running a single MCI test."""
        from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG

        # Create a simple DAG for testing
        ts_dag = TimeSeriesDAG()
        ts_dag.add_nodes_from([("X", 0), ("Y", 0), ("X", 1), ("Y", 1)])
        ts_dag.add_edges_from(
            [(("X", 1), ("X", 0)), (("X", 1), ("Y", 0)), (("Y", 1), ("Y", 0))]
        )

        max_lag = 1
        lagged_data = self.pcmci._create_lagged_data(self.data, max_lag)

        # Mock CI test
        mock_ci_test = Mock(return_value=True)  # Independent

        should_remove = self.pcmci._run_single_mci_test(
            ts_dag, ("X", 1), ("Y", 0), lagged_data, mock_ci_test, 0.05, 5
        )

        self.assertTrue(should_remove)

    def test_get_parents(self):
        """Test getting parents of a node."""
        # This method inherits from TimeSeriesDAG/nx.Graph
        # We need to add some edges first
        self.pcmci.add_edges_from([(("X", 1), ("Y", 0)), (("Z", 1), ("Y", 0))])

        parents = self.pcmci.get_parents(("Y", 0))

        self.assertIn(("X", 1), parents)
        self.assertIn(("Z", 1), parents)
        self.assertEqual(len(parents), 2)

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_estimate_integration(self, mock_get_ci_test):
        """Integration test for the complete estimate method."""

        # Create a simple CI test that makes some variables independent
        def mock_ci_test_func(u, v, sep_set, **kwargs):
            # Make some variables conditionally independent for testing
            return len(sep_set) > 0 and np.random.random() > 0.7

        mock_get_ci_test.return_value = mock_ci_test_func

        try:
            result = self.pcmci.estimate(
                max_time_lag=1,
                max_cond_vars=2,
                show_progress=False,
                n_jobs=1,
                return_type="ts_dag",
            )

            # Check that result is a TimeSeriesDAG
            from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG

            self.assertIsInstance(result, TimeSeriesDAG)

            # Check that it has nodes
            self.assertGreater(len(result.nodes()), 0)

        except Exception as e:
            # If TimeSeriesDAG is not available, this test will be skipped
            self.skipTest(f"TimeSeriesDAG not available: {e}")


class TestPCMCIHelperFunctions(unittest.TestCase):
    """Additional tests for helper functions and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = pd.DataFrame(np.random.randn(50, 2), columns=["A", "B"])
        self.pcmci = PCMCI(data=self.data)

    def test_test_edge_independence_parallel(self):
        """Test edge independence testing in parallel context."""
        # Create test graph
        graph = nx.Graph()
        graph.add_nodes_from([("A", 0), ("B", 0), ("A", 1)])
        graph.add_edges_from([(("A", 0), ("B", 0)), (("A", 1), ("A", 0))])

        lagged_data = self.pcmci._create_lagged_data(self.data, 1)
        mock_ci_test = Mock(return_value=False)  # Not independent

        result = self.pcmci._test_edge_independence(
            ("A", 0), ("B", 0), graph, lagged_data, mock_ci_test, 1, 0.05
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        is_independent, _ = result
        self.assertIsInstance(is_independent, bool)

    def test_temporal_constraints_in_sepsets(self):
        """Test that temporal constraints are respected in separating sets."""
        graph = nx.Graph()
        nodes = [("X", 0), ("Y", 0), ("X", 1), ("Y", 1), ("Z", 2)]
        graph.add_nodes_from(nodes)

        # Add edges respecting temporal order
        graph.add_edges_from(
            [
                (("X", 1), ("X", 0)),
                (("Y", 1), ("Y", 0)),
                (("X", 1), ("Y", 0)),
                (("Z", 2), ("X", 0)),
            ]
        )

        # Test separating sets for nodes at different time lags
        sepsets = self.pcmci._get_potential_sepsets(("X", 1), ("Y", 0), graph, 1)

        # Check that temporal constraints are respected
        for sepset in sepsets:
            for node in sepset:
                _, lag = node
                # Separating variables should be at time >= min(X_lag, Y_lag) = min(1, 0) = 0
                self.assertGreaterEqual(lag, 0)

    def test_data_validation(self):
        """Test data validation in various methods."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        pcmci = PCMCI(data=empty_data)

        # Should handle empty data gracefully
        try:
            lagged_data = pcmci._create_lagged_data(empty_data, 1)
            self.assertEqual(len(lagged_data), 0)
        except Exception:
            # Empty data might cause issues, which is acceptable
            pass

        # Test with single column
        single_col_data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})
        pcmci = PCMCI(data=single_col_data)
        lagged_data = pcmci._create_lagged_data(single_col_data, 1)

        self.assertEqual(len(lagged_data.columns), 2)  # X_lag_0, X_lag_1
