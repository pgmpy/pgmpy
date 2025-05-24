import unittest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import patch, MagicMock

from pgmpy.estimators.PCMCI import PCMCI
from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG


class TestPCMCI(unittest.TestCase):
    def setUp(self):
        """Set up minimal test data."""
        np.random.seed(42)
        self.T = 100  # Reduced from 1000

        # Simple AR(1) process: X_t -> X_{t+1}
        self.ar_data = self._generate_ar_data()
        self.pcmci = PCMCI(
            data=self.ar_data,
        )

        # Bivariate VAR process
        self.var_data = self._generate_var_data()
        self.var_pcmci = PCMCI(data=self.var_data)

    def _generate_ar_data(self):
        """Generate simple AR(1) process."""
        data = pd.DataFrame(np.random.randn(self.T, 1), columns=["X"])
        for t in range(1, self.T):
            data.loc[t, "X"] = 0.7 * data.loc[t - 1, "X"] + 0.1 * np.random.randn()
        return data

    def _generate_var_data(self):
        """Generate bivariate VAR process."""
        data = pd.DataFrame(np.random.randn(self.T, 2), columns=["X", "Y"])
        for t in range(1, self.T):
            data.loc[t, "Y"] += 0.5 * data.loc[t - 1, "X"] + 0.3 * data.loc[t - 1, "Y"]
            data.loc[t, "X"] += 0.4 * data.loc[t - 1, "X"] + 0.1 * np.random.randn()
            data.loc[t, "Y"] += 0.1 * np.random.randn()
        return data

    def test_init(self):
        """Test PCMCI initialization."""
        pcmci = PCMCI(data=self.ar_data)
        self.assertIsNotNone(pcmci.data)

        pcmci_no_data = PCMCI()
        self.assertIsNone(pcmci_no_data.data)

    def test_estimate_basic(self):
        """Test basic estimate functionality."""
        ts_dag = self.pcmci.estimate(
            max_time_lag=2, significance_level=0.1, show_progress=False
        )

        self.assertIsInstance(ts_dag, TimeSeriesDAG)
        self.assertTrue(len(list(ts_dag.nodes())) > 0)

        # Check node structure
        for node in ts_dag.nodes():
            self.assertIsInstance(node, tuple)
            self.assertEqual(len(node), 2)

    def test_estimate_bivariate(self):
        """Test PCMCI on bivariate data."""
        ts_dag = self.var_pcmci.estimate(
            max_time_lag=2, significance_level=0.1, show_progress=False
        )

        self.assertIsInstance(ts_dag, TimeSeriesDAG)
        node_vars = set([node[0] for node in ts_dag.nodes()])
        self.assertTrue({"X", "Y"}.issubset(node_vars))

    def test_estimate_return_types(self):
        """Test different return types."""
        # Test skeleton return
        skeleton, sep_sets = self.pcmci.estimate(
            max_time_lag=2, return_type="skeleton", show_progress=False
        )
        self.assertIsInstance(skeleton, nx.Graph)
        self.assertIsInstance(sep_sets, dict)

        # Test ts_dag return
        ts_dag = self.pcmci.estimate(
            max_time_lag=2, return_type="ts_dag", show_progress=False
        )
        self.assertIsInstance(ts_dag, TimeSeriesDAG)

    def test_create_lagged_data(self):
        """Test lagged data creation."""
        lagged_data = self.pcmci._create_lagged_data(self.ar_data, 2)

        self.assertIsInstance(lagged_data, pd.DataFrame)
        expected_columns = [("X", 0), ("X", 1), ("X", 2)]
        for col in expected_columns:
            self.assertIn(col, lagged_data.columns)

        # Test edge cases
        with self.assertRaises(ValueError):
            self.pcmci._create_lagged_data(self.ar_data, -1)

        with self.assertRaises(TypeError):
            self.pcmci._create_lagged_data(np.array([[1, 2]]), 1)

    def test_build_skeleton(self):
        """Test skeleton building."""
        from pgmpy.estimators.CITests import get_ci_test

        ci_test = get_ci_test("pearsonr", full=True, data=self.ar_data)
        skeleton, sep_sets = self.pcmci._build_time_series_skeleton(
            ci_test=ci_test, max_time_lag=2, significance_level=0.1, show_progress=False
        )

        self.assertIsInstance(skeleton, nx.Graph)
        self.assertIsInstance(sep_sets, dict)

    def test_error_conditions(self):
        """Test error conditions."""
        # No data provided
        pcmci_no_data = PCMCI()
        with self.assertRaises(ValueError):
            pcmci_no_data.estimate(max_time_lag=2)

        # Invalid return type
        with self.assertRaises(ValueError):
            self.pcmci.estimate(
                max_time_lag=2, return_type="invalid", show_progress=False
            )

        # Summary graph not implemented
        with self.assertRaises(NotImplementedError):
            self.pcmci.estimate(
                max_time_lag=2, return_type="summary_graph", show_progress=False
            )

    def test_different_parameters(self):
        """Test with different parameter values."""
        # Different max_time_lag values
        for max_lag in [1, 2]:
            ts_dag = self.pcmci.estimate(max_time_lag=max_lag, show_progress=False)
            self.assertIsInstance(ts_dag, TimeSeriesDAG)

        # Different significance levels
        for sig_level in [0.01, 0.1]:
            ts_dag = self.pcmci.estimate(
                max_time_lag=2, significance_level=sig_level, show_progress=False
            )
            self.assertIsInstance(ts_dag, TimeSeriesDAG)

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_ci_test_integration(self, mock_get_ci_test):
        """Test CI test integration."""
        mock_ci_test = MagicMock(return_value=True)
        mock_get_ci_test.return_value = mock_ci_test

        ts_dag = self.pcmci.estimate(max_time_lag=2, show_progress=False)

        self.assertIsInstance(ts_dag, TimeSeriesDAG)
