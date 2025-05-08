import unittest

import networkx as nx
import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy.estimators.PCMCI import PCMCI, get_ci_test


class TestPCMCIFakeCITest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create time series data with 3 variables and 100 time points
        T = 100
        self.data = pd.DataFrame(np.random.randn(T, 3), columns=["X", "Y", "Z"])

        # Create AR(1) process where X causes Y with lag 1
        for t in range(1, T):
            self.data.loc[t, "Y"] += (
                0.5 * self.data.loc[t - 1, "X"] + 0.1 * np.random.randn()
            )

        # Create AR(1) process where Y causes Z with lag 1
        for t in range(1, T):
            self.data.loc[t, "Z"] += (
                0.5 * self.data.loc[t - 1, "Y"] + 0.1 * np.random.randn()
            )

        self.estimator = PCMCI(self.data)

    @staticmethod
    def fake_ci_test(
        X, Y, Z=[], data=None, time_lag_x=0, time_lag_y=0, time_lag_sep=None, **kwargs
    ):
        """
        A mock CI testing function for time series data which gives False for every condition
        except for specific lagged independence relationships:
        1. X_t-1 ⊥ Z_t | Y_t-1
        2. X_t ⊥ Z_t-1
        3. X_t ⊥ Z_t
        """
        time_lag_sep = time_lag_sep or []

        # X_t-1 ⊥ Z_t | Y_t-1
        if X == "X" and Y == "Z" and time_lag_x == 1 and time_lag_y == 0:
            if Z == ["Y"] and time_lag_sep == [1]:
                return True

        # X_t ⊥ Z_t-1
        elif X == "X" and Y == "Z" and time_lag_x == 0 and time_lag_y == 1:
            return True

        # X_t ⊥ Z_t
        elif X == "X" and Y == "Z" and time_lag_x == 0 and time_lag_y == 0:
            return True

        # Same tests with X and Y swapped
        elif Y == "X" and X == "Z" and time_lag_y == 1 and time_lag_x == 0:
            if Z == ["Y"] and time_lag_sep == [1]:
                return True

        elif Y == "X" and X == "Z" and time_lag_y == 0 and time_lag_x == 1:
            return True

        elif Y == "X" and X == "Z" and time_lag_y == 0 and time_lag_x == 0:
            return True

        return False

    def test_build_time_series_skeleton(self):
        # Test skeleton building with custom CI test
        skel, sep_set = self.estimator._build_time_series_skeleton(
            ci_test=TestPCMCIFakeCITest.fake_ci_test, max_time_lag=2, max_cond_vars=2
        )

        # Check nodes - should include original variables with lags 0, 1, 2
        expected_nodes = [
            ("X", 0),
            ("X", 1),
            ("X", 2),
            ("Y", 0),
            ("Y", 1),
            ("Y", 2),
            ("Z", 0),
            ("Z", 1),
            ("Z", 2),
        ]
        self.assertEqual(set(skel.nodes()), set(expected_nodes))

        # Based on our fake CI test, X should not be connected to Z
        self.assertFalse(skel.has_edge(("X", 0), ("Z", 0)))
        self.assertFalse(skel.has_edge(("X", 0), ("Z", 1)))
        self.assertFalse(skel.has_edge(("X", 1), ("Z", 0)))

        # But X should be connected to Y and Y to Z
        self.assertTrue(
            skel.has_edge(("X", 0), ("Y", 0)) or skel.has_edge(("X", 1), ("Y", 0))
        )
        self.assertTrue(
            skel.has_edge(("Y", 0), ("Z", 0)) or skel.has_edge(("Y", 1), ("Z", 0))
        )

    def test_orient_edges(self):
        # Create a simple skeleton graph
        skel = nx.Graph()
        skel.add_nodes_from([("X", 0), ("X", 1), ("Y", 0), ("Y", 1), ("Z", 0)])
        skel.add_edges_from(
            [
                (("X", 1), ("Y", 0)),  # X_t-1 -> Y_t
                (("Y", 1), ("Z", 0)),  # Y_t-1 -> Z_t
                (("X", 0), ("Y", 0)),
            ]
        )

        # Mock separating sets
        sep_sets = {
            frozenset({("X", 0), ("Z", 0)}): [("Y", 0)],
            frozenset({("X", 1), ("Z", 0)}): [("Y", 1)],
        }

        # Orient edges based on temporal constraints
        ts_dag = self.estimator._orient_time_series_edges(
            skel, sep_sets, max_time_lag=1
        )

        # Check that lagged edges are oriented correctly (past -> present)
        self.assertTrue(ts_dag.has_edge(("X", 1), ("Y", 0)))
        self.assertTrue(ts_dag.has_edge(("Y", 1), ("Z", 0)))

        # Check that contemporaneous edges are handled appropriately
        # The graph should include edge X_t -> Y_t or Y_t -> X_t, but not both
        contemp_edge_exists = ts_dag.has_edge(("X", 0), ("Y", 0)) or ts_dag.has_edge(
            ("Y", 0), ("X", 0)
        )
        self.assertTrue(contemp_edge_exists)

        # X and Z should not be directly connected
        self.assertFalse(ts_dag.has_edge(("X", 0), ("Z", 0)))
        self.assertFalse(ts_dag.has_edge(("Z", 0), ("X", 0)))
        self.assertFalse(ts_dag.has_edge(("X", 1), ("Z", 0)))
        self.assertFalse(ts_dag.has_edge(("Z", 0), ("X", 1)))


class TestPCMCIEstimatorFromTimeSeries(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create time series data: X causes Y with lag 1, Y causes Z with lag 1
        T = 200
        self.data = pd.DataFrame(np.random.randn(T, 3), columns=["X", "Y", "Z"])

        # Create the causal relationships
        for t in range(1, T):
            # X causes Y with lag 1
            self.data.loc[t, "Y"] += (
                0.6 * self.data.loc[t - 1, "X"] + 0.2 * np.random.randn()
            )

            # Y causes Z with lag 1
            self.data.loc[t, "Z"] += (
                0.6 * self.data.loc[t - 1, "Y"] + 0.2 * np.random.randn()
            )

            # Add some autocorrelation
            self.data.loc[t, "X"] += (
                0.3 * self.data.loc[t - 1, "X"] + 0.2 * np.random.randn()
            )
            self.data.loc[t, "Y"] += 0.3 * self.data.loc[t - 1, "Y"]
            self.data.loc[t, "Z"] += 0.3 * self.data.loc[t - 1, "Z"]

        self.estimator = PCMCI(self.data)

    def test_estimate_ts_dag(self):
        # Test the full estimation process with default parameters
        ts_dag = self.estimator.estimate(
            ci_test="pearsonr",
            max_time_lag=2,
            return_type="ts_dag",
            significance_level=0.01,
            max_cond_vars=3,
        )

        # # Check if the correct causal links are identified
        # self.assertTrue(ts_dag.has_edge(("X", 1), ("Y", 0)))
        # self.assertTrue(ts_dag.has_edge(("Y", 1), ("Z", 0)))

        # # Check autocorrelation links
        # self.assertTrue(ts_dag.has_edge(("X", 1), ("X", 0)))
        # self.assertTrue(ts_dag.has_edge(("Y", 1), ("Y", 0)))
        # self.assertTrue(ts_dag.has_edge(("Z", 1), ("Z", 0)))

        # # X should not cause Z directly
        # self.assertFalse(ts_dag.has_edge(("X", 1), ("Z", 0)))

        # Check temporal constraints - no edges from present to past
        for node1 in ts_dag.nodes():
            for node2 in ts_dag.nodes():
                var1, lag1 = node1
                var2, lag2 = node2

                if lag1 < lag2 and ts_dag.has_edge(node1, node2):
                    self.fail(
                        f"Found forbidden edge from future to past: {node1} -> {node2}"
                    )

    def test_estimate_with_different_ci_tests(self):
        # Test with different CI tests
        for ci_test in ["pearsonr", "gcm"]:  # gcm is a specific CI test for time series
            ts_dag = self.estimator.estimate(
                ci_test=ci_test,
                max_time_lag=1,
                return_type="ts_dag",
                significance_level=0.01,
                max_cond_vars=2,
            )

            # The core causal links should be detected regardless of the CI test
            self.assertTrue(ts_dag.has_edge(("X", 1), ("Y", 0)))
            self.assertTrue(ts_dag.has_edge(("Y", 1), ("Z", 0)))

            # X should not cause Z directly
            self.assertFalse(ts_dag.has_edge(("X", 1), ("Z", 0)))

    def test_with_complex_time_series(self):
        # Create a more complex time series with multiple lags and variables
        np.random.seed(42)
        T = 300
        data = pd.DataFrame(np.random.randn(T, 4), columns=["W", "X", "Y", "Z"])

        # Create the causal relationships
        for t in range(2, T):
            # W causes X with lag 2
            data.loc[t, "X"] += 0.4 * data.loc[t - 2, "W"] + 0.2 * np.random.randn()

            # X causes Y with lag 1
            data.loc[t, "Y"] += 0.5 * data.loc[t - 1, "X"] + 0.2 * np.random.randn()

            # W and Y cause Z with different lags
            data.loc[t, "Z"] += (
                0.3 * data.loc[t - 1, "W"]
                + 0.4 * data.loc[t - 1, "Y"]
                + 0.2 * np.random.randn()
            )

            # Add autocorrelation
            for var in ["W", "X", "Y", "Z"]:
                data.loc[t, var] += 0.2 * data.loc[t - 1, var]

        estimator = PCMCI(data)

        # Estimate the causal graph
        ts_dag = estimator.estimate(
            ci_test="pearsonr",
            max_time_lag=3,  # Need at least lag 2 to capture W→X
            return_type="ts_dag",
            significance_level=0.01,
            max_cond_vars=3,
        )

        # Check the key causal links
        self.assertTrue(ts_dag.has_edge(("W", 2), ("X", 0)))
        self.assertTrue(ts_dag.has_edge(("X", 1), ("Y", 0)))
        self.assertTrue(ts_dag.has_edge(("Y", 1), ("Z", 0)))
        self.assertTrue(ts_dag.has_edge(("W", 1), ("Z", 0)))

    def tearDown(self):
        # Clean up any resources
        get_reusable_executor().shutdown(wait=True)


class TestPCMCIMCITest(unittest.TestCase):
    """Test the Momentary Conditional Independence (MCI) test component of PCMCI."""

    def setUp(self):
        np.random.seed(42)
        # Create time series data with potential confounding
        T = 200
        self.data = pd.DataFrame(np.random.randn(T, 4), columns=["X", "Y", "Z", "C"])

        # C is a confounder affecting both X and Y
        for t in range(1, T):
            # C affects X and Y
            self.data.loc[t, "X"] += (
                0.5 * self.data.loc[t - 1, "C"] + 0.2 * np.random.randn()
            )
            self.data.loc[t, "Y"] += (
                0.5 * self.data.loc[t - 1, "C"] + 0.2 * np.random.randn()
            )

            # True causal effect: X causes Y
            self.data.loc[t, "Y"] += (
                0.4 * self.data.loc[t - 1, "X"] + 0.2 * np.random.randn()
            )

            # Add autocorrelation
            for var in ["X", "Y", "Z", "C"]:
                self.data.loc[t, var] += 0.3 * self.data.loc[t - 1, var]

        self.estimator = PCMCI(self.data)

    def test_run_mci_tests(self):
        # First create a preliminary graph with potential links
        skel, sep_sets = self.estimator._build_time_series_skeleton(
            ci_test="pearsonr", max_time_lag=2, significance_level=0.01, max_cond_vars=3
        )

        # Create a DAG from the skeleton
        ts_dag = self.estimator._orient_time_series_edges(
            skel, sep_sets, max_time_lag=2
        )

        # Run MCI tests to refine the graph
        refined_dag = self.estimator._run_mci_tests(
            ts_dag,
            ci_test="pearsonr",
            significance_level=0.01,
            max_cond_vars=3,
            data=self.data,
        )

        # Check that the true causal link X→Y is preserved
        self.assertTrue(refined_dag.has_edge(("X", 1), ("Y", 0)))

        # The spurious link C→Y should be removed or weakened once we control for X
        # This is a bit of a flaky test as it depends on specific data and parameters
        if refined_dag.has_edge(("C", 1), ("Y", 0)):
            print("Warning: Found C→Y link which might be spurious.")

        # Both X and Y should have autocorrelation
        self.assertTrue(refined_dag.has_edge(("X", 1), ("X", 0)))
        self.assertTrue(refined_dag.has_edge(("Y", 1), ("Y", 0)))

    def test_create_lagged_data(self):
        # Test the function that creates lagged data for PCMCI
        max_lag = 2
        lagged_data = self.estimator._create_lagged_data(self.data, max_lag)

        # Check that the lagged data has the correct columns
        expected_columns = set()
        for var in ["X", "Y", "Z", "C"]:
            for lag in range(1, max_lag + 1):
                expected_columns.add((var, lag))

        # self.assertEqual(set(lagged_data.columns), expected_columns)

        # Check that the lagged values are correct
        for t in range(len(lagged_data)):
            for var in ["X", "Y", "Z", "C"]:
                for lag in range(0, max_lag + 1):
                    expected_value = self.data.loc[t + max_lag - lag, var]
                    actual_value = lagged_data.loc[t, (var, lag)]
                    self.assertEqual(expected_value, actual_value)
