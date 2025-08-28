import unittest

import numpy as np

from pgmpy.base.AncestralBase import AncestralBase


class TestAncestralBase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.graph = AncestralBase()

    def test_init_empty(self):
        """Test initialization with no edges."""
        graph = AncestralBase()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)

    def test_init_with_edges(self):
        """Test initialization with edge list."""
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        graph = AncestralBase(ebunch=edges)
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 2)

    def test_add_edge_valid(self):
        """Test adding valid edges with different mark combinations."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("B", "C", ">", "-")
        self.graph.add_edge("C", "D", "o", "o")

        self.assertEqual(len(self.graph.edges), 3)
        self.assertIn("A", self.graph.nodes)
        self.assertIn("B", self.graph.nodes)
        self.assertIn("C", self.graph.nodes)
        self.assertIn("D", self.graph.nodes)

    def test_add_edge_same_node_error(self):
        """Test that adding edge with same source and target raises error."""
        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "A", "-", ">")

    def test_add_edge_invalid_marks_error(self):
        """Test that invalid marks raise ValueError."""
        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "B", "x", ">")
        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "B", "-", "y")
        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "B", "z", "w")

    def test_add_edges_from(self):
        """Test adding multiple edges from list."""
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("A", "C", "o", "o")]
        self.graph.add_edges_from(edges)

        self.assertEqual(len(self.graph.edges), 3)
        self.assertEqual(len(self.graph.nodes), 3)

    def test_get_neighbors_basic(self):
        """Test getting neighbors without constraints."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("A", "C", ">", "-")

        neighbors_a = self.graph.get_neighbors("A")
        self.assertEqual(neighbors_a, {"B", "C"})

        neighbors_b = self.graph.get_neighbors("B")
        self.assertEqual(neighbors_b, {"A"})

    def test_get_neighbors_nonexistent_node(self):
        """Test getting neighbors for non-existent node."""
        neighbors = self.graph.get_neighbors("Z")
        self.assertEqual(neighbors, set())

    def test_get_parents(self):
        """Test getting parent nodes."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("C", "B", "-", ">")
        self.graph.add_edge("B", "D", "-", ">")

        parents_b = self.graph.get_parents("B")
        self.assertEqual(parents_b, {"A", "C"})

        parents_d = self.graph.get_parents("D")
        self.assertEqual(parents_d, {"B"})

        parents_a = self.graph.get_parents("A")
        self.assertEqual(parents_a, set())

    def test_get_children(self):
        """Test getting child nodes."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("A", "C", "-", ">")
        self.graph.add_edge("B", "D", "-", ">")

        children_a = self.graph.get_children("A")
        self.assertEqual(children_a, {"B", "C"})

        children_b = self.graph.get_children("B")
        self.assertEqual(children_b, {"D"})

        children_d = self.graph.get_children("D")
        self.assertEqual(children_d, set())

    def test_get_spouses(self):
        """Test getting spouse nodes (bidirectional arrows)."""
        self.graph.add_edge("A", "B", ">", ">")
        self.graph.add_edge("A", "C", "-", ">")
        self.graph.add_edge("C", "D", ">", ">")

        spouses_a = self.graph.get_spouses("A")
        self.assertEqual(spouses_a, {"B"})

        spouses_c = self.graph.get_spouses("C")
        self.assertEqual(spouses_c, {"D"})

        spouses_b = self.graph.get_spouses("B")
        self.assertEqual(spouses_b, {"A"})

    def test_get_ancestors(self):
        """Test getting all ancestors."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("B", "C", "-", ">")
        self.graph.add_edge("C", "D", "-", ">")
        self.graph.add_edge("E", "C", "-", ">")

        ancestors_d = self.graph.get_ancestors("D")
        self.assertEqual(ancestors_d, {"A", "B", "C", "D", "E"})

        ancestors_c = self.graph.get_ancestors("C")
        self.assertEqual(ancestors_c, {"A", "B", "C", "E"})

        ancestors_a = self.graph.get_ancestors("A")
        self.assertEqual(ancestors_a, set("A"))

    def test_get_descendants(self):
        """Test getting all descendants."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("B", "C", "-", ">")
        self.graph.add_edge("C", "D", "-", ">")
        self.graph.add_edge("B", "E", "-", ">")

        descendants_a = self.graph.get_descendants("A")
        self.assertEqual(descendants_a, {"A", "B", "C", "D", "E"})

        descendants_b = self.graph.get_descendants("B")
        self.assertEqual(descendants_b, {"B", "C", "D", "E"})

        descendants_d = self.graph.get_descendants("D")
        self.assertEqual(descendants_d, set("D"))

    def test_get_reachable_nodes(self):
        """Test getting reachable nodes with constraints."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("B", "C", "-", ">")
        self.graph.add_edge("A", "D", "o", "o")
        self.graph.add_edge("D", "E", "o", "o")

        reachable_directed = self.graph.get_reachable_nodes("A", v_type=">")
        self.assertEqual(reachable_directed, {"A", "B", "C"})

        reachable_undirected = self.graph.get_reachable_nodes("A", v_type="o")
        self.assertEqual(reachable_undirected, {"A", "D", "E"})

    def test_adjacency_matrix(self):
        """Test conversion to adjacency matrix."""
        self.graph.add_edge("A", "B", "-", ">")
        self.graph.add_edge("B", "C", ">", "-")

        M, node_index = self.graph.adjacency_matrix

        self.assertEqual(M.shape, (3, 3))
        self.assertEqual(len(node_index), 3)

        self.assertIn("A", node_index)
        self.assertIn("B", node_index)
        self.assertIn("C", node_index)

        a_idx, b_idx, c_idx = node_index["A"], node_index["B"], node_index["C"]

        self.assertEqual(M[a_idx, b_idx], ">")
        self.assertEqual(M[b_idx, a_idx], "-")

        self.assertEqual(M[b_idx, c_idx], "-")
        self.assertEqual(M[c_idx, b_idx], ">")

        self.assertEqual(M[a_idx, c_idx], "")
        self.assertEqual(M[c_idx, a_idx], "")

    def test_adjacency_matrix_empty_graph(self):
        """Test adjacency matrix for empty graph."""
        M, node_index = self.graph.adjacency_matrix
        self.assertEqual(M.shape, (0, 0))
        self.assertEqual(len(node_index), 0)

    def test_adjacency_matrix_setter(self):
        """Test setting graph from adjacency matrix."""
        M = np.array([[0, ">", 0], ["-", 0, ">"], [0, "-", 0]], dtype=object)

        self.graph.adjacency_matrix = M

        self.assertEqual(len(self.graph.nodes), 3)
        self.assertEqual(len(self.graph.edges), 2)

        self.assertTrue(self.graph.has_edge("X_0", "X_1"))
        self.assertTrue(self.graph.has_edge("X_1", "X_2"))

        self.assertEqual(self.graph["X_0"]["X_1"]["marks"], {"X_0": ">", "X_1": "-"})
        self.assertEqual(self.graph["X_1"]["X_2"]["marks"], {"X_1": ">", "X_2": "-"})
