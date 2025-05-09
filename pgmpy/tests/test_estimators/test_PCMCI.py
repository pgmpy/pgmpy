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
        pass

    def tearDown(self):
        # Clean up any resources
        get_reusable_executor().shutdown(wait=True)
