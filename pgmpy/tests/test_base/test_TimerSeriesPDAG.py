import unittest

from pgmpy.base import TimeSeriesDAG, TimeSeriesPDAG


class TestTimeSeriesPDAG(unittest.TestCase):
    def setUp(self):
        self.ts_pdag_mix = TimeSeriesPDAG(
            directed_ebunch=[("A_0", "C_0"), ("D_0", "C_1")],
            undirected_ebunch=[("B_0", "A_0"), ("B_0", "D_1")],
            num_time_slices=2,
        )
        self.ts_pdag_dir = TimeSeriesPDAG(
            directed_ebunch=[
                ("A_0", "B_0"),
                ("D_0", "B_1"),
                ("A_0", "C_1"),
                ("D_0", "C_0"),
            ],
            num_time_slices=2,
        )
        self.ts_pdag_undir = TimeSeriesPDAG(
            undirected_ebunch=[
                ("A_0", "C_0"),
                ("D_0", "C_1"),
                ("B_0", "A_0"),
                ("B_0", "D_1"),
            ],
            num_time_slices=2,
        )
        self.ts_pdag_latent = TimeSeriesPDAG(
            directed_ebunch=[("A_0", "C_0"), ("D_0", "C_1")],
            undirected_ebunch=[("B_0", "A_0"), ("B_0", "D_1")],
            latents=["A_0", "D_0"],
            num_time_slices=2,
        )

    def test_init_normal(self):
        # Mix directed and undirected
        directed_edges = [("A_0", "C_0"), ("D_0", "C_1")]
        undirected_edges = [("B_0", "A_0"), ("B_0", "D_1")]
        pdag = TimeSeriesPDAG(
            directed_ebunch=directed_edges,
            undirected_ebunch=undirected_edges,
            num_time_slices=2,
        )
        expected_edges = {
            ("A_0", "C_0"),
            ("D_0", "C_1"),
            ("A_0", "B_0"),
            ("B_0", "A_0"),
            ("B_0", "D_1"),
            ("D_1", "B_0"),
        }
        self.assertEqual(set(pdag.edges()), expected_edges)
        self.assertEqual(set(pdag.nodes()), {"A_0", "B_0", "C_0", "D_0", "C_1", "D_1"})
        self.assertEqual(pdag.directed_edges, set(directed_edges))
        self.assertEqual(pdag.undirected_edges, set(undirected_edges))
        self.assertEqual(pdag.num_time_slices, 2)

    def test_add_temporal_edge_directed(self):
        pdag = TimeSeriesPDAG(num_time_slices=2)
        pdag.add_nodes_from(["A_0", "B_0", "A_1", "B_1"])

        # Add temporal directed edge within same time slice
        pdag.add_temporal_directed_edge("A", "B", time_slice=0)
        self.assertTrue(("A_0", "B_0") in pdag.edges())
        self.assertTrue(("A_0", "B_0") in pdag.directed_edges)

        # Add temporal directed edge between time slices
        pdag.add_temporal_directed_edge("A", "B", from_time_slice=0, to_time_slice=1)
        self.assertTrue(("A_0", "B_1") in pdag.edges())
        self.assertTrue(("A_0", "B_1") in pdag.directed_edges)

    def test_add_temporal_edge_undirected(self):
        pdag = TimeSeriesPDAG(num_time_slices=2)
        pdag.add_nodes_from(["A_0", "B_0", "A_1", "B_1"])

        # Add temporal undirected edge within same time slice
        pdag.add_temporal_undirected_edge("A", "B", time_slice=0)
        self.assertTrue(("A_0", "B_0") in pdag.edges())
        self.assertTrue(("B_0", "A_0") in pdag.edges())
        self.assertTrue(("A_0", "B_0") in pdag.undirected_edges)

        # Add temporal undirected edge between time slices
        pdag.add_temporal_undirected_edge("A", "B", from_time_slice=0, to_time_slice=1)
        self.assertTrue(("A_0", "B_1") in pdag.edges())
        self.assertTrue(("B_1", "A_0") in pdag.edges())
        self.assertTrue(("A_0", "B_1") in pdag.undirected_edges)

    def test_get_intra_slice_edges(self):
        pdag = TimeSeriesPDAG(
            directed_ebunch=[("A_0", "B_0"), ("B_1", "C_1")],
            undirected_ebunch=[("C_0", "D_0"), ("D_1", "E_1")],
            num_time_slices=2,
        )

        intra_slice_directed = pdag.get_intra_slice_directed_edges()
        self.assertEqual(set(intra_slice_directed), {("A_0", "B_0"), ("B_1", "C_1")})

        intra_slice_undirected = pdag.get_intra_slice_undirected_edges()
        self.assertEqual(set(intra_slice_undirected), {("C_0", "D_0"), ("D_1", "E_1")})

    def test_get_inter_slice_edges(self):
        pdag = TimeSeriesPDAG(
            directed_ebunch=[("A_0", "B_0"), ("A_0", "A_1")],
            undirected_ebunch=[("C_0", "D_0"), ("C_0", "C_1")],
            num_time_slices=2,
        )

        inter_slice_directed = pdag.get_inter_slice_directed_edges()
        self.assertEqual(set(inter_slice_directed), {("A_0", "A_1")})

        inter_slice_undirected = pdag.get_inter_slice_undirected_edges()
        self.assertEqual(set(inter_slice_undirected), {("C_0", "C_1")})

    def test_get_slice(self):
        pdag = TimeSeriesPDAG(
            directed_ebunch=[("A_0", "B_0"), ("C_1", "D_1")],
            undirected_ebunch=[("B_0", "C_0"), ("D_1", "E_1")],
            num_time_slices=2,
        )

        # Get slice as PDAG
        slice_0 = pdag.get_slice(0)
        self.assertEqual(set(slice_0.nodes()), {"A_0", "B_0", "C_0"})
        self.assertEqual(slice_0.directed_edges, {("A_0", "B_0")})
        self.assertEqual(slice_0.undirected_edges, {("B_0", "C_0")})

        slice_1 = pdag.get_slice(1)
        self.assertEqual(set(slice_1.nodes()), {"C_1", "D_1", "E_1"})
        self.assertEqual(slice_1.directed_edges, {("C_1", "D_1")})
        self.assertEqual(slice_1.undirected_edges, {("D_1", "E_1")})

    def test_to_time_series_dag(self):
        pdag = self.ts_pdag_mix

        # Convert PDAG to DAG
        dag = pdag.to_time_series_dag()

        # Check that it's a DAG
        self.assertIsInstance(dag, TimeSeriesDAG)
        self.assertEqual(dag.num_time_slices, 2)

        # Check that directed edges are preserved
        for edge in pdag.directed_edges:
            self.assertTrue(edge in dag.edges())

        # Check total edge count (all edges oriented now)
        self.assertEqual(len(dag.edges()), 4)

    def test_latent_variables(self):
        pdag = self.ts_pdag_latent

        # Check latents are preserved
        self.assertEqual(pdag.latents, set(["A_0", "D_0"]))

        # Convert to DAG and check latents are preserved
        dag = pdag.to_time_series_dag()
        self.assertEqual(dag.latents, set(["A_0", "D_0"]))

    def tearDown(self):
        del self.ts_pdag_mix
        del self.ts_pdag_dir
        del self.ts_pdag_undir
        del self.ts_pdag_latent
