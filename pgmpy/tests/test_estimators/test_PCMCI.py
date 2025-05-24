#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import Mock, patch

from pgmpy.estimators import PCMCI


class TestPCMCI:
    """Minimal test suite for PCMCI class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        T = 100
        data = pd.DataFrame(
            {"X": np.random.randn(T), "Y": np.random.randn(T), "Z": np.random.randn(T)}
        )

        # Add some temporal dependencies
        for t in range(1, T):
            data.loc[t, "Y"] += 0.5 * data.loc[t - 1, "X"] + 0.3 * data.loc[t - 1, "Y"]
            data.loc[t, "Z"] += 0.4 * data.loc[t - 1, "Y"]

        return data

    @pytest.fixture
    def pcmci_instance(self, sample_data):
        """Create PCMCI instance with sample data."""
        return PCMCI(data=sample_data)

    def test_initialization(self, sample_data):
        """Test PCMCI initialization."""
        pcmci = PCMCI(data=sample_data)
        assert pcmci.data is not None
        assert pcmci.data.equals(sample_data)
        assert pcmci.independencies is None

    def test_initialization_without_data(self):
        """Test PCMCI initialization without data."""
        pcmci = PCMCI()
        assert pcmci.data is None
        assert pcmci.independencies is None

    def test_estimate_without_data_raises_error(self):
        """Test that estimate raises error when no data is provided."""
        pcmci = PCMCI()
        with pytest.raises(ValueError, match="Data must be provided"):
            pcmci.estimate()

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_estimate_returns_ts_dag(self, mock_ci_test, pcmci_instance):
        """Test that estimate returns a TimeSeriesDAG by default."""
        # Mock the CI test to return a simple function
        mock_ci_test.return_value = Mock(return_value=False)  # Always dependent

        result = pcmci_instance.estimate(max_time_lag=1, max_cond_vars=2)

        # Should return a TimeSeriesDAG object
        assert hasattr(result, "nodes")
        assert hasattr(result, "edges")

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_estimate_returns_skeleton(self, mock_ci_test, pcmci_instance):
        """Test that estimate can return skeleton when requested."""
        mock_ci_test.return_value = Mock(return_value=False)

        skeleton, separating_sets = pcmci_instance.estimate(
            max_time_lag=1, return_type="skeleton"
        )

        assert isinstance(skeleton, nx.Graph)
        assert isinstance(separating_sets, dict)

    def test_estimate_invalid_return_type(self, pcmci_instance):
        """Test that invalid return_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid return_type"):
            pcmci_instance.estimate(return_type="invalid")

    def test_estimate_summary_graph_not_implemented(self, pcmci_instance):
        """Test that summary_graph return_type raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Summary graph is not yet implemented"
        ):
            pcmci_instance.estimate(return_type="summary_graph")

    def test_create_lagged_data(self, pcmci_instance):
        """Test _create_lagged_data method."""
        data = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [10, 20, 30, 40, 50]})

        lagged_data = pcmci_instance._create_lagged_data(data, max_lag=2)

        # Check that lagged columns exist
        expected_columns = [("X", 0), ("X", 1), ("X", 2), ("Y", 0), ("Y", 1), ("Y", 2)]
        assert all(col in lagged_data.columns for col in expected_columns)

        # Check that NaN rows are dropped
        assert not lagged_data.isnull().any().any()

        # Check shape (should lose max_lag rows)
        assert len(lagged_data) == len(data) - 2

    def test_create_lagged_data_invalid_input(self, pcmci_instance):
        """Test _create_lagged_data with invalid inputs."""
        # Non-DataFrame input
        with pytest.raises(TypeError, match="Data must be a pandas DataFrame"):
            pcmci_instance._create_lagged_data([1, 2, 3], max_lag=1)

        # Negative max_lag
        data = pd.DataFrame({"X": [1, 2, 3]})
        with pytest.raises(ValueError, match="max_lag must be non-negative"):
            pcmci_instance._create_lagged_data(data, max_lag=-1)

    def test_get_potential_sepsets(self, pcmci_instance):
        """Test _get_potential_sepsets method."""
        # Create a simple graph
        graph = nx.Graph()
        nodes = [("X", 0), ("Y", 0), ("Z", 1), ("W", 1)]
        graph.add_nodes_from(nodes)
        graph.add_edges_from(
            [(("X", 0), ("Z", 1)), (("Y", 0), ("Z", 1)), (("X", 0), ("W", 1))]
        )

        # Test getting potential separating sets
        sepsets = pcmci_instance._get_potential_sepsets(
            ("X", 0), ("Y", 0), graph, cond_set_size=1
        )

        # Should return list of tuples
        assert isinstance(sepsets, list)
        # Each sepset should be a combination of nodes
        for sepset in sepsets:
            assert isinstance(sepset, tuple)

    @patch("pgmpy.estimators.CITests.get_ci_test")
    def test_build_time_series_skeleton(self, mock_ci_test, pcmci_instance):
        """Test _build_time_series_skeleton method."""
        # Mock CI test to always return independence
        mock_ci_test.return_value = Mock(return_value=True)

        skeleton, sep_sets = pcmci_instance._build_time_series_skeleton(
            ci_test=mock_ci_test.return_value,
            max_time_lag=1,
            max_cond_vars=2,
            show_progress=False,
        )

        assert isinstance(skeleton, nx.Graph)
        assert isinstance(sep_sets, dict)

        # Check that time-lagged nodes are created
        expected_nodes = [("X", 0), ("X", 1), ("Y", 0), ("Y", 1), ("Z", 0), ("Z", 1)]
        for node in expected_nodes:
            assert node in skeleton.nodes()

    def test_orient_time_series_edges(self, pcmci_instance):
        """Test _orient_time_series_edges method."""
        # Create a simple skeleton
        skeleton = nx.Graph()
        skeleton.add_edges_from(
            [
                (("X", 1), ("Y", 0)),  # X at t-1 -> Y at t
                (("Y", 1), ("Y", 0)),  # Y at t-1 -> Y at t (autocorrelation)
            ]
        )

        separating_sets = {}

        ts_dag = pcmci_instance._orient_time_series_edges(
            skeleton, separating_sets, max_time_lag=1
        )

        # Check that edges are oriented correctly (past -> present)
        assert ts_dag.has_edge(("X", 1), ("Y", 0))
        assert ts_dag.has_edge(("Y", 1), ("Y", 0))

        # Check that reverse edges don't exist
        assert not ts_dag.has_edge(("Y", 0), ("X", 1))
        assert not ts_dag.has_edge(("Y", 0), ("Y", 1))

    def test_test_edge_independence(self, pcmci_instance):
        """Test _test_edge_independence method."""
        # Create simple graph and data
        graph = nx.Graph()
        graph.add_edge(("X", 0), ("Y", 0))

        data = pd.DataFrame({("X", 0): [1, 2, 3], ("Y", 0): [4, 5, 6]})

        # Mock CI test
        mock_ci_test = Mock(return_value=True)  # Independent

        is_independent, sep_set = pcmci_instance._test_edge_independence(
            ("X", 0),
            ("Y", 0),
            graph,
            data,
            mock_ci_test,
            cond_set_size=0,
            significance_level=0.05,
        )

        assert isinstance(is_independent, bool)
        # sep_set could be None or a list
        assert sep_set is None or isinstance(sep_set, (list, tuple))

    def test_run_single_mci_test(self, pcmci_instance):
        """Test _run_single_mci_test method."""
        # Create simple DAG
        from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG

        ts_dag = TimeSeriesDAG()
        ts_dag.add_edge(("X", 1), ("Y", 0))

        data = pd.DataFrame(
            {("X", 0): [1, 2, 3], ("X", 1): [4, 5, 6], ("Y", 0): [7, 8, 9]}
        )

        # Mock CI test
        mock_ci_test = Mock(return_value=False)  # Dependent

        should_remove = pcmci_instance._run_single_mci_test(
            ts_dag,
            ("X", 1),
            ("Y", 0),
            data,
            mock_ci_test,
            significance_level=0.05,
            max_cond_vars=2,
        )

        assert isinstance(should_remove, bool)

    def test_remove_cycles_within_time_slice(self, pcmci_instance):
        """Test _remove_cycles_within_time_slice method."""
        from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG

        ts_dag = TimeSeriesDAG()

        # Add nodes and create a cycle within the same time slice
        ts_dag.add_edge(("X", 0), ("Y", 0))
        ts_dag.add_edge(("Y", 0), ("X", 0))

        initial_edges = list(ts_dag.edges())
        pcmci_instance._remove_cycles_within_time_slice(ts_dag)
        final_edges = list(ts_dag.edges())

        # Should have fewer edges after cycle removal
        assert len(final_edges) < len(initial_edges)

    def test_get_parents_method_exists(self, pcmci_instance):
        """Test that get_parents method exists and is callable."""
        # This is a basic test since the method relies on TimeSeriesDAG functionality
        assert hasattr(pcmci_instance, "get_parents")
        assert callable(pcmci_instance.get_parents)


# Additional integration test
class TestPCMCIIntegration:
    """Integration tests for PCMCI with realistic scenarios."""

    def test_simple_ar_process(self):
        """Test PCMCI on a simple AR process."""
        np.random.seed(42)
        T = 200

        # Generate AR(1) process: Y(t) = 0.5 * Y(t-1) + noise
        data = pd.DataFrame(np.random.randn(T, 2), columns=["X", "Y"])
        for t in range(1, T):
            data.loc[t, "Y"] = 0.5 * data.loc[t - 1, "Y"] + 0.1 * np.random.randn()

        pcmci = PCMCI(data=data)

        # Should not raise errors
        try:
            with patch("pgmpy.estimators.CITests.get_ci_test") as mock_ci_test:
                mock_ci_test.return_value = Mock(return_value=False)
                result = pcmci.estimate(max_time_lag=2, show_progress=False)
                assert result is not None
        except Exception as e:
            pytest.fail(f"PCMCI estimation failed with error: {e}")
