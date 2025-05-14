import unittest

import networkx as nx
import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy.estimators.PCMCI import PCMCI


class TestPCMCIFullEstimation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create time series data with 3 variables and 200 time points
        T = 200
        self.data = pd.DataFrame(np.random.randn(T, 3), columns=["X", "Y", "Z"])

        # Create AR(1) process where X causes Y with lag 1
        for t in range(1, T):
            self.data.loc[t, "Y"] += (
                0.6 * self.data.loc[t - 1, "X"] + 0.2 * np.random.randn()
            )

        # Create AR(1) process where Y causes Z with lag 1
        for t in range(1, T):
            self.data.loc[t, "Z"] += (
                0.6 * self.data.loc[t - 1, "Y"] + 0.2 * np.random.randn()
            )

        # Add autocorrelation
        for t in range(1, T):
            self.data.loc[t, "X"] += 0.3 * self.data.loc[t - 1, "X"]
            self.data.loc[t, "Y"] += 0.3 * self.data.loc[t - 1, "Y"]
            self.data.loc[t, "Z"] += 0.3 * self.data.loc[t - 1, "Z"]

        self.estimator = PCMCI(self.data)

    def test_estimate_full_pipeline(self):
        """Test the full PCMCI estimation pipeline."""
        # Run the estimation with default settings
        ts_dag = self.estimator.estimate(max_time_lag=2, significance_level=0.05)

        # Test basic properties of the resulting graph
        self.assertIsInstance(ts_dag, nx.DiGraph)

        # Check for expected causal relationships
        # X(t-1) should cause Y(t)
        self.assertTrue(ts_dag.has_edge(("X", 1), ("Y", 0)))

        # Y(t-1) should cause Z(t)
        self.assertTrue(ts_dag.has_edge(("Y", 1), ("Z", 0)))

        # Autocorrelation edges
        self.assertTrue(ts_dag.has_edge(("X", 1), ("X", 0)))
        self.assertTrue(ts_dag.has_edge(("Y", 1), ("Y", 0)))
        self.assertTrue(ts_dag.has_edge(("Z", 1), ("Z", 0)))

        # X should not directly cause Z (it's indirect through Y)
        self.assertFalse(ts_dag.has_edge(("X", 1), ("Z", 0)))

    def test_estimate_with_different_ci_tests(self):
        """Test estimation with different CI test options."""
        # Test with g_sq (G-square) test
        ts_dag_gsq = self.estimator.estimate(ci_test="g_sq", max_time_lag=1)
        self.assertIsInstance(ts_dag_gsq, nx.DiGraph)

        # Test with chi_square test
        ts_dag_chi = self.estimator.estimate(ci_test="chi_square", max_time_lag=1)
        self.assertIsInstance(ts_dag_chi, nx.DiGraph)

        # Test with pearsonr test
        ts_dag_pearson = self.estimator.estimate(ci_test="pearsonr", max_time_lag=1)
        self.assertIsInstance(ts_dag_pearson, nx.DiGraph)

    def test_estimate_with_parallel_vs_sequential(self):
        """Test estimation with parallel vs sequential execution."""
        # Run with sequential execution
        ts_dag_seq = self.estimator.estimate(max_time_lag=1, n_jobs=1)

        # Run with parallel execution
        ts_dag_par = self.estimator.estimate(max_time_lag=1, n_jobs=2)

        # The results should be topologically equivalent
        # Check that the same edges exist
        self.assertEqual(set(ts_dag_seq.edges()), set(ts_dag_par.edges()))

    def test_estimate_return_types(self):
        """Test different return types from the estimate method."""
        # Test returning skeleton
        skeleton, sep_sets = self.estimator.estimate(
            max_time_lag=1, return_type="skeleton"
        )
        self.assertIsInstance(skeleton, nx.Graph)
        self.assertIsInstance(sep_sets, dict)

        # Test returning ts_dag (default)
        ts_dag = self.estimator.estimate(max_time_lag=1, return_type="ts_dag")
        self.assertIsInstance(ts_dag, nx.DiGraph)

        # Test for invalid return type
        with self.assertRaises(ValueError):
            self.estimator.estimate(max_time_lag=1, return_type="invalid")

    def tearDown(self):
        # Clean up any resources
        get_reusable_executor().shutdown(wait=True)


class TestPCMCIEdgeCases(unittest.TestCase):
    def test_empty_data(self):
        """Test handling of empty data."""
        # Create empty dataframe
        empty_data = pd.DataFrame()

        estimator = PCMCI()  # No data provided

        # Should raise ValueError when trying to estimate
        with self.assertRaises(ValueError):
            estimator.estimate()

    def test_single_variable(self):
        """Test with single variable time series."""
        # Create single variable time series
        T = 100
        data = pd.DataFrame(np.random.randn(T, 1), columns=["X"])

        # Add autocorrelation
        for t in range(1, T):
            data.loc[t, "X"] += 0.5 * data.loc[t - 1, "X"]

        estimator = PCMCI(data)
        ts_dag = estimator.estimate(max_time_lag=2)

        # There should only be autocorrelation edges
        self.assertTrue(ts_dag.has_edge(("X", 1), ("X", 0)))
        self.assertTrue(
            ts_dag.has_edge(("X", 2), ("X", 0)) or ts_dag.has_edge(("X", 2), ("X", 1))
        )

        # Only 3 nodes for X with lags 0, 1, 2
        self.assertEqual(len(ts_dag.nodes()), 3)

    def test_max_cond_vars_limit(self):
        """Test behavior when max_cond_vars is small."""
        np.random.seed(42)
        # Create time series with 5 variables
        T = 100
        data = pd.DataFrame(np.random.randn(T, 5), columns=["A", "B", "C", "D", "E"])

        # Make them all causally related
        for t in range(1, T):
            data.loc[t, "B"] += 0.4 * data.loc[t - 1, "A"]
            data.loc[t, "C"] += 0.4 * data.loc[t - 1, "B"]
            data.loc[t, "D"] += 0.4 * data.loc[t - 1, "C"]
            data.loc[t, "E"] += 0.4 * data.loc[t - 1, "D"]

        estimator = PCMCI(data)

        # Set max_cond_vars to a small value
        ts_dag = estimator.estimate(max_time_lag=1, max_cond_vars=1)

        # With such constraints, should still find core causal chain
        self.assertTrue(ts_dag.has_edge(("A", 1), ("B", 0)))
        self.assertTrue(ts_dag.has_edge(("B", 1), ("C", 0)))
        self.assertTrue(ts_dag.has_edge(("C", 1), ("D", 0)))
        self.assertTrue(ts_dag.has_edge(("D", 1), ("E", 0)))

    def test_high_significance_level(self):
        """Test with a high significance level (more edges)."""
        np.random.seed(42)
        T = 100
        data = pd.DataFrame(np.random.randn(T, 3), columns=["X", "Y", "Z"])

        # X causes Y
        for t in range(1, T):
            data.loc[t, "Y"] += 0.3 * data.loc[t - 1, "X"]

        estimator = PCMCI(data)

        # With high significance level
        ts_dag_high = estimator.estimate(max_time_lag=1, significance_level=0.5)

        # With low significance level
        ts_dag_low = estimator.estimate(max_time_lag=1, significance_level=0.01)

        # Higher significance should lead to more edges
        self.assertGreaterEqual(len(ts_dag_high.edges()), len(ts_dag_low.edges()))

    def tearDown(self):
        # Clean up any resources
        get_reusable_executor().shutdown(wait=True)


class TestPCMCIMCIPhase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Complex scenario: X → Y → Z and X → Z
        T = 200
        self.data = pd.DataFrame(np.random.randn(T, 3), columns=["X", "Y", "Z"])

        # X causes Y with lag 1
        for t in range(1, T):
            self.data.loc[t, "Y"] += 0.6 * self.data.loc[t - 1, "X"]

        # Y causes Z with lag 1
        for t in range(1, T):
            self.data.loc[t, "Z"] += 0.4 * self.data.loc[t - 1, "Y"]

        # X also causes Z with lag 2 (direct path)
        for t in range(2, T):
            self.data.loc[t, "Z"] += 0.3 * self.data.loc[t - 2, "X"]

        self.estimator = PCMCI(self.data)

    def test_run_mci_tests(self):
        """Test the MCI test phase directly."""
        # First build a skeleton and orient it
        skeleton, sep_sets = self.estimator._build_time_series_skeleton(
            ci_test="pearsonr", max_time_lag=2
        )

        ts_dag = self.estimator._orient_time_series_edges(
            skeleton, sep_sets, max_time_lag=2
        )

        # Now run MCI tests
        refined_dag = self.estimator._run_mci_tests(
            ts_dag, ci_test="pearsonr", significance_level=0.05
        )

        # The refined DAG should show X->Y->Z and also X->Z
        self.assertTrue(refined_dag.has_edge(("X", 1), ("Y", 0)))
        self.assertTrue(refined_dag.has_edge(("Y", 1), ("Z", 0)))
        self.assertTrue(refined_dag.has_edge(("X", 2), ("Z", 0)))

    def test_run_single_mci_test(self):
        """Test a single MCI test directly."""
        # Create a mock DAG
        ts_dag = nx.DiGraph()
        ts_dag.add_nodes_from([("X", 0), ("X", 1), ("Y", 0), ("Y", 1), ("Z", 0)])
        ts_dag.add_edges_from(
            [
                (("X", 1), ("Y", 0)),
                (("Y", 1), ("Z", 0)),
            ]
        )

        # Create lagged data
        lagged_data = self.estimator._create_lagged_data(self.data, 2)

        # Test an edge that should remain
        should_remove = self.estimator._run_single_mci_test(
            ts_dag,
            ("X", 1),
            ("Y", 0),
            lagged_data,
            ci_test="pearsonr",
            significance_level=0.05,
            max_cond_vars=3,
        )

        self.assertFalse(should_remove)  # X->Y should remain

        # Test an edge that might be removed (X->Z) might be found redundant
        # Add this edge first
        ts_dag.add_edge(("X", 1), ("Z", 0))

        should_remove = self.estimator._run_single_mci_test(
            ts_dag,
            ("X", 1),
            ("Z", 0),
            lagged_data,
            ci_test="pearsonr",
            significance_level=0.05,
            max_cond_vars=3,
        )

        # Not testing the result since it's data dependent,
        # but testing the function runs correctly

    def test_get_parents(self):
        """Test the get_parents method."""
        # Setup a graph with known parents
        ts_dag = nx.DiGraph()
        ts_dag.add_nodes_from([("X", 0), ("X", 1), ("Y", 0), ("Y", 1), ("Z", 0)])
        ts_dag.add_edges_from(
            [
                (("X", 1), ("Y", 0)),
                (("Y", 1), ("Z", 0)),
                (("X", 1), ("Z", 0)),
            ]
        )

        # Associate the graph with the estimator
        self.estimator.add_nodes_from(ts_dag.nodes())
        self.estimator.add_edges_from(ts_dag.edges())

        # Test getting parents
        z_parents = self.estimator.get_parents(("Z", 0))
        self.assertEqual(set(z_parents), {("X", 1), ("Y", 1)})

        y_parents = self.estimator.get_parents(("Y", 0))
        self.assertEqual(set(y_parents), {("X", 1)})

        x0_parents = self.estimator.get_parents(("X", 0))
        self.assertEqual(set(x0_parents), set())

    def tearDown(self):
        # Clean up any resources
        get_reusable_executor().shutdown(wait=True)


class TestPCMCICycles(unittest.TestCase):
    def test_remove_cycles_within_time_slice(self):
        """Test removing cycles within the same time slice."""
        # Create a graph with a cycle within the same time slice
        ts_dag = nx.DiGraph()
        ts_dag.add_nodes_from(
            [
                ("X", 0),
                ("Y", 0),
                ("Z", 0),  # Time slice 0
                ("X", 1),
                ("Y", 1),
                ("Z", 1),  # Time slice 1
            ]
        )

        # Add a cycle in time slice 0
        ts_dag.add_edge(("X", 0), ("Y", 0))
        ts_dag.add_edge(("Y", 0), ("Z", 0))
        ts_dag.add_edge(("Z", 0), ("X", 0))  # This creates a cycle

        # Add normal time-lagged edges
        ts_dag.add_edge(("X", 1), ("X", 0))
        ts_dag.add_edge(("Y", 1), ("Y", 0))

        # Create an estimator
        estimator = PCMCI()

        # Remove cycles
        estimator._remove_cycles_within_time_slice(ts_dag)

        # There should be no cycles in the time slice 0
        subgraph = ts_dag.subgraph([("X", 0), ("Y", 0), ("Z", 0)])
        self.assertFalse(list(nx.simple_cycles(subgraph)))

        # Should have broken one of the edges in the cycle
        edges_count = sum(
            [
                ts_dag.has_edge(("X", 0), ("Y", 0)),
                ts_dag.has_edge(("Y", 0), ("Z", 0)),
                ts_dag.has_edge(("Z", 0), ("X", 0)),
            ]
        )
        self.assertEqual(edges_count, 2)  # One edge was removed
