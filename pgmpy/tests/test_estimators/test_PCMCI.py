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
        """Test the full PCMCI estimation pipeline with actual time series data."""
        # Use a lower significance level to be more permissive in edge detection
        ts_dag = self.estimator.estimate(
            ci_test="pearsonr",
            significance_level=0.05,
            max_time_lag=2,
            max_cond_vars=3,
            show_progress=False,
        )

        # True causal relationships in our data:
        # 1. X(t-1) -> X(t) [autocorrelation]
        # 2. Y(t-1) -> Y(t) [autocorrelation]
        # 3. Z(t-1) -> Z(t) [autocorrelation]
        # 4. X(t-1) -> Y(t) [causal]
        # 5. Y(t-1) -> Z(t) [causal]

        # Check that the known causal edges exist
        # Note: Use explicit tuple comparisons to avoid array comparison issues

        # Check autocorrelation edges
        self.assertTrue(
            any(u == ("X", 1) and v == ("X", 0) for u, v in ts_dag.edges()),
            "Missing autocorrelation edge X(t-1) -> X(t)",
        )
        self.assertTrue(
            any(u == ("Y", 1) and v == ("Y", 0) for u, v in ts_dag.edges()),
            "Missing autocorrelation edge Y(t-1) -> Y(t)",
        )
        self.assertTrue(
            any(u == ("Z", 1) and v == ("Z", 0) for u, v in ts_dag.edges()),
            "Missing autocorrelation edge Z(t-1) -> Z(t)",
        )

        # Check causal edges
        self.assertTrue(
            any(u == ("X", 1) and v == ("Y", 0) for u, v in ts_dag.edges()),
            "Missing causal edge X(t-1) -> Y(t)",
        )
        self.assertTrue(
            any(u == ("Y", 1) and v == ("Z", 0) for u, v in ts_dag.edges()),
            "Missing causal edge Y(t-1) -> Z(t)",
        )

        # Check that implausible edges do NOT exist (X should not directly affect Z)
        self.assertFalse(
            any(u == ("X", 1) and v == ("Z", 0) for u, v in ts_dag.edges()),
            "Should not have direct edge X(t-1) -> Z(t)",
        )

        # Verify no edges from future to past exist (temporal constraint)
        for u, v in ts_dag.edges():
            _, lag_u = u
            _, lag_v = v
            self.assertFalse(
                lag_u < lag_v, f"Edge from {u} to {v} violates temporal constraints"
            )

    def test_create_lagged_data(self):
        """Test the creation of lagged data for time series analysis."""
        max_lag = 2
        lagged_data = self.estimator._create_lagged_data(self.data, max_lag)

        # Check that the lagged data has the right columns
        expected_columns = []
        for var in ["X", "Y", "Z"]:
            for lag in range(max_lag + 1):
                expected_columns.append((var, lag))

        self.assertEqual(set(lagged_data.columns), set(expected_columns))

        # Check that the number of rows is correct (original rows - max_lag)
        expected_rows = len(self.data) - max_lag
        self.assertEqual(len(lagged_data), expected_rows)

        # Check a specific value to ensure correct alignment
        orig_value = self.data.loc[max_lag, "X"]  # Value at t=max_lag
        lagged_value = lagged_data.loc[
            0, ("X", 0)
        ]  # First row in lagged data for X at lag 0
        self.assertAlmostEqual(orig_value, lagged_value)

    def tearDown(self):
        # Clean up any resources
        get_reusable_executor().shutdown(wait=True)
