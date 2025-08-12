import unittest
from itertools import combinations
from unittest.mock import Mock, patch

import networkx as nx
import pandas as pd

from pgmpy.base import UndirectedGraph
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator
from pgmpy.independencies import Independencies


class TestBaseConstraintEstimator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.sample_data = pd.DataFrame(
            {
                "A": [0, 1, 0, 1, 0, 1, 0, 1],
                "B": [0, 0, 1, 1, 0, 0, 1, 1],
                "C": [0, 1, 1, 0, 1, 0, 0, 1],
                "D": [1, 1, 0, 0, 1, 1, 0, 0],
            }
        )

        self.sample_independencies = Independencies(["A", "B"], ["C", "D"])

        self.estimator = BaseConstraintEstimator(data=self.sample_data)

    def test_init_with_data(self):
        """Test initialization with data."""
        estimator = BaseConstraintEstimator(data=self.sample_data)
        self.assertIsNotNone(estimator.data)
        self.assertEqual(list(estimator.data.columns), ["A", "B", "C", "D"])

    def test_init_with_independencies(self):
        """Test initialization with independencies."""
        estimator = BaseConstraintEstimator(independencies=self.sample_independencies)
        self.assertIsNone(estimator.data)
        self.assertIsNotNone(estimator.independencies)

    def test_get_potential_sepsets_basic(self):
        """Test _get_potential_sepsets with basic graph."""
        graph = UndirectedGraph()
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        sepsets = list(self.estimator._get_potential_sepsets("A", "B", {}, graph, 1))
        expected_sepsets = [("C",)]
        self.assertEqual(sepsets, expected_sepsets)

    def test_get_potential_sepsets_complete_graph(self):
        """Test _get_potential_sepsets with complete graph."""
        graph = nx.complete_graph(["A", "B", "C", "D"], create_using=nx.Graph)

        sepsets = list(self.estimator._get_potential_sepsets("A", "B", {}, graph, 1))

        # A and B both have neighbors C, D (excluding each other)
        # Should get combinations of size 1 from both sets
        expected_count = (
            4  # ('C',), ('D',) from A neighbors + ('C',), ('D',) from B neighbors
        )
        self.assertEqual(len(sepsets), expected_count)

    def test_get_potential_sepsets_with_temporal_ordering(self):
        """Test _get_potential_sepsets with temporal ordering."""
        graph = nx.complete_graph(["A", "B", "C", "D"], create_using=nx.Graph)
        temporal_ordering = {"A": 1, "B": 2, "C": 3, "D": 4}

        sepsets = list(
            self.estimator._get_potential_sepsets("A", "B", temporal_ordering, graph, 1)
        )

        # Only nodes with temporal order <= min(1, 2) = 1 should be considered
        # Only A has order 1, but A and B exclude each other
        self.assertEqual(sepsets, [])

    def test_get_potential_sepsets_no_neighbors(self):
        """Test _get_potential_sepsets when nodes have no other neighbors."""
        graph = UndirectedGraph()
        graph.add_edge("A", "B")

        sepsets = list(self.estimator._get_potential_sepsets("A", "B", {}, graph, 1))

        self.assertEqual(sepsets, [])

    @patch("pgmpy.estimators.CITests.get_callable_ci_test")
    def test_build_skeleton_no_independence(self, mock_get_ci_test):
        """Test skeleton building when no independence found."""
        mock_ci_test = Mock(return_value=False)
        mock_get_ci_test.return_value = mock_ci_test

        graph, separating_sets = self.estimator.build_skeleton(
            max_cond_vars=1, show_progress=False
        )

        # Should return complete graph when no independence found
        expected_edges = len(list(combinations(self.sample_data.columns, 2)))
        self.assertEqual(len(graph.edges()), expected_edges)
        self.assertEqual(len(separating_sets), 0)

    @patch("pgmpy.estimators.CITests.get_callable_ci_test")
    def test_build_skeleton_with_independence(self, mock_get_ci_test):
        """Test skeleton building when independence found."""

        def mock_ci_test_func(u, v, conditioning_set, **kwargs):
            # Make A and B independent with empty conditioning set
            return u == "A" and v == "B" and len(conditioning_set) == 0

        mock_get_ci_test.return_value = mock_ci_test_func

        graph, separating_sets = self.estimator.build_skeleton(
            max_cond_vars=1, show_progress=False
        )

        # Edge (A, B) should be removed
        self.assertFalse(graph.has_edge("A", "B"))
        self.assertIn(frozenset(["A", "B"]), separating_sets)
        self.assertEqual(separating_sets[frozenset(["A", "B"])], ())

    @patch("pgmpy.estimators.CITests.get_callable_ci_test")
    def test_build_skeleton_max_cond_vars(self, mock_get_ci_test):
        """Test skeleton building respects max_cond_vars limit."""
        mock_ci_test = Mock(return_value=False)
        mock_get_ci_test.return_value = mock_ci_test

        with patch("pgmpy.global_vars.logger") as mock_logger:
            graph, _ = self.estimator.build_skeleton(
                max_cond_vars=0, show_progress=False  # Very low limit
            )

            # Should reach the limit and log message
            mock_logger.info.assert_called_with(
                "Reached maximum number of allowed conditional variables. Exiting"
            )

    @patch("pgmpy.estimators.CITests.get_callable_ci_test")
    def test_build_skeleton_return_types(self, mock_get_ci_test):
        """Test build_skeleton returns correct types."""
        mock_ci_test = Mock(return_value=False)
        mock_get_ci_test.return_value = mock_ci_test

        result = self.estimator.build_skeleton(max_cond_vars=1, show_progress=False)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        graph, separating_sets = result
        self.assertIsInstance(graph, nx.Graph)
        self.assertIsInstance(separating_sets, dict)

    def test_build_skeleton_separating_sets_format(self):
        """Test separating sets are in correct format."""
        with patch("pgmpy.estimators.CITests.get_callable_ci_test") as mock_get_ci_test:

            def mock_ci_test_func(u, v, conditioning_set, **kwargs):
                return u == "A" and v == "B" and conditioning_set == ("C",)

            mock_get_ci_test.return_value = mock_ci_test_func

            graph, separating_sets = self.estimator.build_skeleton(
                max_cond_vars=1, show_progress=False
            )

            # Check format of separating sets
            for edge_set, sep_set in separating_sets.items():
                self.assertIsInstance(edge_set, frozenset)
                self.assertEqual(len(edge_set), 2)
                self.assertIsInstance(sep_set, tuple)

    def tearDown(self):
        """Clean up test fixtures."""
        del self.sample_data
        del self.sample_independencies
        del self.estimator
