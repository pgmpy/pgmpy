import unittest
import warnings

import numpy as np
import pandas as pd

import pgmpy.tests.help_functions as hf
from pgmpy.base import TimeSeriesDAG


class TestTimeSeriesDAGCreation(unittest.TestCase):
    def setUp(self):
        self.graph = TimeSeriesDAG()

    def test_class_init_without_data(self):
        self.assertIsInstance(self.graph, TimeSeriesDAG)
        self.assertEqual(self.graph.num_time_slices, 1)

    def test_class_init_with_time_series_edges(self):
        # Test with init data
        self.graph = TimeSeriesDAG([("a_0", "b_0"), ("b_0", "c_1"), ("c_0", "c_1")], num_time_slices=2)
        self.assertListEqual(sorted(self.graph.nodes()), ["a_0", "b_0", "c_0", "c_1"])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()), [["a_0", "b_0"], ["b_0", "c_1"], ["c_0", "c_1"]]
        )
        self.assertEqual(self.graph.num_time_slices, 2)
        self.assertEqual(self.graph.latents, set())

        # Test with latent variables
        self.graph = TimeSeriesDAG([("a_0", "b_0"), ("b_0", "c_1"), ("c_0", "c_1")], 
                                  num_time_slices=2, latents=["b_0"])
        self.assertEqual(self.graph.latents, set(["b_0"]))

    def test_add_edge_with_time_slice(self):
        self.graph = TimeSeriesDAG(num_time_slices=3)
        self.graph.add_edge("a_0", "b_0")
        self.graph.add_edge("b_0", "c_1")
        self.graph.add_edge("c_1", "d_2")
        
        self.assertListEqual(sorted(self.graph.nodes()), ["a_0", "b_0", "c_1", "d_2"])
        self.assertListEqual(
            hf.recursive_sorted(self.graph.edges()), [["a_0", "b_0"], ["b_0", "c_1"], ["c_1", "d_2"]]
        )
        
    def test_add_node_with_time_slice(self):
        self.graph = TimeSeriesDAG(num_time_slices=3)
        self.graph.add_node("a_0")
        self.graph.add_node("b_1")
        self.graph.add_node("c_2")
        
        self.assertListEqual(sorted(self.graph.nodes()), ["a_0", "b_1", "c_2"])
        
    def test_add_temporal_edge(self):
        self.graph = TimeSeriesDAG(num_time_slices=2)
        self.graph.add_nodes_from(["a_0", "b_0", "a_1", "b_1"])
        
        # Add temporal edge within same time slice
        self.graph.add_temporal_edge("a", "b", time_slice=0)
        self.assertTrue(("a_0", "b_0") in self.graph.edges())
        
        # Add temporal edge between time slices
        self.graph.add_temporal_edge("a", "b", from_time_slice=0, to_time_slice=1)
        self.assertTrue(("a_0", "b_1") in self.graph.edges())
        
    def test_add_temporal_edges_from(self):
        self.graph = TimeSeriesDAG(num_time_slices=2)
        self.graph.add_nodes_from(["a_0", "b_0", "c_0", "a_1", "b_1", "c_1"])
        
        # Add temporal edges within same time slice
        self.graph.add_temporal_edges_from([("a", "b"), ("b", "c")], time_slice=0)
        self.assertTrue(("a_0", "b_0") in self.graph.edges())
        self.assertTrue(("b_0", "c_0") in self.graph.edges())
        
        # Add temporal edges between time slices
        self.graph.add_temporal_edges_from([("a", "b"), ("b", "c")], 
                                         from_time_slice=0, to_time_slice=1)
        self.assertTrue(("a_0", "b_1") in self.graph.edges())
        self.assertTrue(("b_0", "c_1") in self.graph.edges())
        
    def test_get_intra_slice_edges(self):
        self.graph = TimeSeriesDAG([("a_0", "b_0"), ("b_0", "c_0"), 
                                  ("a_0", "a_1"), ("b_0", "b_1")], num_time_slices=2)
        
        intra_slice = self.graph.get_intra_slice_edges()
        self.assertEqual(set(intra_slice), {("a_0", "b_0"), ("b_0", "c_0")})
        
    def test_get_inter_slice_edges(self):
        self.graph = TimeSeriesDAG([("a_0", "b_0"), ("b_0", "c_0"), 
                                  ("a_0", "a_1"), ("b_0", "b_1")], num_time_slices=2)
        
        inter_slice = self.graph.get_inter_slice_edges()
        self.assertEqual(set(inter_slice), {("a_0", "a_1"), ("b_0", "b_1")})
        
    def test_get_slice(self):
        self.graph = TimeSeriesDAG([("a_0", "b_0"), ("b_0", "c_0"), 
                                  ("a_1", "b_1"), ("b_1", "c_1"),
                                  ("a_0", "a_1"), ("b_0", "b_1")], num_time_slices=2)
        
        # Get specific time slice
        slice_0 = self.graph.get_slice(0)
        self.assertEqual(set(slice_0.nodes()), {"a_0", "b_0", "c_0"})
        self.assertEqual(set(slice_0.edges()), {("a_0", "b_0"), ("b_0", "c_0")})
        
        slice_1 = self.graph.get_slice(1)
        self.assertEqual(set(slice_1.nodes()), {"a_1", "b_1", "c_1"})
        self.assertEqual(set(slice_1.edges()), {("a_1", "b_1"), ("b_1", "c_1")})
        
    def test_temporal_markov_blanket(self):
        self.graph = TimeSeriesDAG([
            ("a_0", "b_0"), ("b_0", "c_0"),
            ("a_1", "b_1"), ("b_1", "c_1"),
            ("a_0", "a_1"), ("b_0", "b_1"), ("c_0", "c_1")
        ], num_time_slices=2)
        
        # Get temporal Markov blanket for node in time slice 0
        mb_a0 = self.graph.get_temporal_markov_blanket("a_0")
        self.assertEqual(set(mb_a0), {"b_0", "a_1"})
        
        # Get temporal Markov blanket for node in time slice 1
        mb_b1 = self.graph.get_temporal_markov_blanket("b_1")
        self.assertEqual(set(mb_b1), {"a_1", "c_1", "b_0"})
        
    def test_temporal_induced_graph(self):
        # Test creating a 2-time-slice Bayesian network (2-TBN)
        self.graph = TimeSeriesDAG(num_time_slices=3)
        
        # Add intra-slice edges for time slice 0
        self.graph.add_temporal_edges_from([("a", "b"), ("b", "c")], time_slice=0)
        
        # Add intra-slice edges for time slice 1
        self.graph.add_temporal_edges_from([("a", "b"), ("b", "c")], time_slice=1)
        
        # Add intra-slice edges for time slice 2
        self.graph.add_temporal_edges_from([("a", "b"), ("b", "c")], time_slice=2)
        
        # Add inter-slice edges from time slice 0 to time slice 1
        self.graph.add_temporal_edges_from([("a", "a"), ("b", "b"), ("c", "c")],
                                         from_time_slice=0, to_time_slice=1)
        
        # Add inter-slice edges from time slice 1 to time slice 2
        self.graph.add_temporal_edges_from([("a", "a"), ("b", "b"), ("c", "c")],
                                         from_time_slice=1, to_time_slice=2)
        
        # Extract the 2-TBN 
        two_tbn = self.graph.get_temporal_induced_graph(from_time_slice=0, to_time_slice=1)
        
        expected_nodes = {"a_0", "b_0", "c_0", "a_1", "b_1", "c_1"}
        expected_edges = {
            ("a_0", "b_0"), ("b_0", "c_0"),
            ("a_1", "b_1"), ("b_1", "c_1"),
            ("a_0", "a_1"), ("b_0", "b_1"), ("c_0", "c_1")
        }
        
        self.assertEqual(set(two_tbn.nodes()), expected_nodes)
        self.assertEqual(set(two_tbn.edges()), expected_edges)
        
    def test_time_slice_template(self):
        # Test creating a DAG template and expanding to multiple time slices
        template = TimeSeriesDAG()
        template.add_edges_from([("a", "b"), ("b", "c")])
        
        # Create temporal model with 3 time slices
        temporal_model = template.to_time_series_dag(num_time_slices=3, 
                                                  temporal_edges=[("a", "a"), ("b", "b")])
        
        expected_nodes = {
            "a_0", "b_0", "c_0", "a_1", "b_1", "c_1", "a_2", "b_2", "c_2"
        }
        expected_edges = {
            # Intra-slice edges for time slice 0
            ("a_0", "b_0"), ("b_0", "c_0"),
            # Intra-slice edges for time slice 1
            ("a_1", "b_1"), ("b_1", "c_1"),
            # Intra-slice edges for time slice 2
            ("a_2", "b_2"), ("b_2", "c_2"),
            # Inter-slice edges from time slice 0 to 1
            ("a_0", "a_1"), ("b_0", "b_1"),
            # Inter-slice edges from time slice 1 to 2
            ("a_1", "a_2"), ("b_1", "b_2")
        }
        
        self.assertEqual(set(temporal_model.nodes()), expected_nodes)
        self.assertEqual(set(temporal_model.edges()), expected_edges)
        
    def tearDown(self):
        del self.graph