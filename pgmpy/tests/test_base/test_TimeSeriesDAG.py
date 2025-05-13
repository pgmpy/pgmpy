import unittest
import warnings

import numpy as np
import pandas as pd

import pgmpy.tests.help_functions as hf
from pgmpy.base.TimeSeriesDAG import TimeSeriesDAG


class TestTimeSeriesDAGCreation(unittest.TestCase):
    def setUp(self):
        self.graph = TimeSeriesDAG()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, TimeSeriesDAG)
        self.assertEqual(self.graph.num_time_slices, 1)

    def test_class_init_with_time_series_edges(self):
        self.graph = TimeSeriesDAG(
            [(("a", 0), ("b", 0)), (("b", 0), ("c", 1)), (("c", 0), ("c", 1))],
            num_time_slices=2,
        )
        self.assertListEqual(
            sorted(self.graph.nodes()), [("a", 0), ("b", 0), ("c", 0), ("c", 1)]
        )
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()),
            [[("a", 0), ("b", 0)], [("b", 0), ("c", 1)], [("c", 0), ("c", 1)]],
        )
        self.assertEqual(self.graph.num_time_slices, 2)
        self.assertEqual(self.graph.latents, set())

        self.graph = TimeSeriesDAG(
            [(("a", 0), ("b", 0)), (("b", 0), ("c", 1)), (("c", 0), ("c", 1))],
            num_time_slices=2,
            latents=[("b", 0)],
        )
        self.assertEqual(self.graph.latents, {("b", 0)})

    def test_add_edge_with_time_slice(self):
        self.graph = TimeSeriesDAG(
            ebunch=[(("a", 0), ("b", 0)), (("b", 0), ("c", 1)), (("c", 1), ("d", 2))],
            num_time_slices=3,
        )
        self.assertListEqual(
            sorted(self.graph.nodes()), [("a", 0), ("b", 0), ("c", 1), ("d", 2)]
        )
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()),
            [[("a", 0), ("b", 0)], [("b", 0), ("c", 1)], [("c", 1), ("d", 2)]],
        )

    def test_add_node_with_time_slice(self):
        self.graph = TimeSeriesDAG(num_time_slices=3)
        self.graph.add_node(("a", 0))
        self.graph.add_node(("b", 1))
        self.graph.add_node(("c", 2))
        self.assertListEqual(sorted(self.graph.nodes()), [("a", 0), ("b", 1), ("c", 2)])

    def test_plot_summary_graph(self):
        import matplotlib.pyplot as plt

        # Create a simple TimeSeriesDAG
        self.graph = TimeSeriesDAG(
            ebunch=[(("a", 0), ("b", 1)), (("b", 1), ("c", 2))],
            num_time_slices=3,
        )

        # Plot the summary graph
        fig, ax = self.graph.plot_summary_graph()

        # Check that the plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        graph = TimeSeriesDAG(
            ebunch=[(("a", 0), ("b", 1), ("c", 2), ("d", 1))],
            num_time_slices=3,
        )

    def tearDown(self):
        del self.graph
