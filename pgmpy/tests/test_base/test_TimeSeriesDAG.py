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

    # def test_add_temporal_edge(self):
    #     self.graph = TimeSeriesDAG(num_time_slices=2)
    #     self.graph.add_temporal_edge("a", "b", time_slice=0)
    #     self.assertIn((("a", 0), ("b", 0)), self.graph.edges())

    # self.graph.add_temporal_edge("a", "b", from_time_slice=0, to_time_slice=1)
    # self.assertIn((("a", 0), ("b", 1)), self.graph.edges())

    # def test_temporal_markov_blanket(self):
    #     self.graph = TimeSeriesDAG(
    #         [
    #             (("a", 0), ("b", 0)),
    #             (("b", 0), ("c", 0)),
    #             (("a", 1), ("b", 1)),
    #             (("b", 1), ("c", 1)),
    #             (("a", 0), ("a", 1)),
    #             (("b", 0), ("b", 1)),
    #             (("c", 0), ("c", 1)),
    #         ],
    #         num_time_slices=2,
    #     )
    #     mb_a0 = self.graph.get_temporal_markov_blanket(("a", 0))
    #     self.assertEqual(set(mb_a0), {("b", 0), ("a", 1)})

    #     mb_b1 = self.graph.get_temporal_markov_blanket(("b", 1))
    #     self.assertEqual(set(mb_b1), {("a", 1), ("c", 1), ("b", 0)})

    def tearDown(self):
        del self.graph
