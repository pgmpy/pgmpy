import unittest

import numpy as np

from pgmpy.base.AncestralBase import AncestralBase


class TestAncestralBase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        edges = [
            ("A", "B", "-", ">"),
            ("A", "C", ">", "-"),
            ("A", "D", "o", "o"),
            ("A", "E", ">", ">"),
            ("A", "F", "-", "-"),
            ("A", "G", "-", "o"),
            ("A", "H", "o", "-"),
            ("A", "I", "o", ">"),
            ("A", "J", ">", "o"),
            ("B", "X", "-", ">"),
            ("C", "Y", ">", "-"),
        ]
        self.graph = AncestralBase(ebunch=edges)

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
        self.assertEqual(graph["A"]["B"]["marks"], {"A": "-", "B": ">"})
        self.assertEqual(graph["B"]["C"]["marks"], {"B": ">", "C": "-"})

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
        graph = AncestralBase()
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("A", "C", "o", "o")]
        graph.add_edges_from(edges)

        self.assertEqual(len(graph.edges), 3)
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(graph["A"]["B"]["marks"], {"A": "-", "B": ">"})
        self.assertEqual(graph["B"]["C"]["marks"], {"B": ">", "C": "-"})
        self.assertEqual(graph["A"]["C"]["marks"], {"A": "o", "C": "o"})

    def test_get_neighbors_basic(self):
        """Test getting neighbors without constraints."""
        all_nodes_except_A = {"B", "C", "D", "E", "F", "G", "H", "I", "J"}
        self.assertEqual(self.graph.get_neighbors("A"), all_nodes_except_A)
        self.assertEqual(self.graph.get_neighbors("B"), {"A", "X"})
        self.assertEqual(self.graph.get_neighbors("C"), {"A", "Y"})

    def test_get_neighbors_nonexistent_node(self):
        """Test getting neighbors for non-existent node."""
        self.assertEqual(self.graph.get_neighbors("Z"), set())

    def test_get_parents(self):
        """Test getting parent nodes."""
        self.assertEqual(self.graph.get_parents("A"), {"C"})
        self.assertEqual(self.graph.get_parents("B"), {"A"})
        self.assertEqual(self.graph.get_parents("C"), {"Y"})

    def test_get_children(self):
        """Test getting child nodes."""
        self.assertEqual(self.graph.get_children("A"), {"B"})
        self.assertEqual(self.graph.get_children("B"), {"X"})
        self.assertEqual(self.graph.get_children("Y"), {"C"})

    def test_get_spouses(self):
        """Test getting spouse nodes (bidirectional arrows)."""
        self.assertEqual(self.graph.get_spouses("A"), {"E"})
        self.assertEqual(self.graph.get_spouses("E"), {"A"})
        self.assertEqual(self.graph.get_spouses("B"), set())

    def test_get_ancestors(self):
        """Test getting all ancestors."""
        self.assertEqual(self.graph.get_ancestors("A"), {"A", "C", "Y"})
        self.assertEqual(self.graph.get_ancestors("X"), {"X", "B", "A", "C", "Y"})

    def test_get_descendants(self):
        """Test getting all descendants."""
        self.assertEqual(self.graph.get_descendants("A"), {"A", "B", "X"})
        self.assertEqual(self.graph.get_descendants("Y"), {"Y", "C", "A", "B", "X"})

    def test_get_reachable_nodes(self):
        """Test getting reachable nodes with constraints."""
        self.assertEqual(
            self.graph.get_reachable_nodes("A", v_type=">"), {"A", "B", "X", "E", "I"}
        )
        self.assertEqual(
            self.graph.get_reachable_nodes("A", u_type=">", v_type="-"), {"A", "C", "Y"}
        )
        self.assertEqual(
            self.graph.get_reachable_nodes("A", u_type="o", v_type="o"), {"A", "D"}
        )
        self.assertEqual(
            self.graph.get_reachable_nodes("A", u_type=">", v_type=">"), {"A", "E"}
        )
        self.assertEqual(
            self.graph.get_reachable_nodes("A", u_type="-", v_type="-"), {"A", "F"}
        )
        self.assertEqual(
            self.graph.get_reachable_nodes("A", u_type="o", v_type=">"), {"A", "I"}
        )
        self.assertEqual(
            self.graph.get_reachable_nodes("A", u_type=">", v_type="o"), {"A", "J"}
        )

    def test_adjacency_matrix(self):
        """Test conversion to adjacency matrix."""
        edges = [
            ("A", "B", "-", ">"),
            ("B", "C", ">", "-"),
        ]
        graph = AncestralBase(ebunch=edges)
        M, node_index = graph.adjacency_matrix

        expected = np.array([[0, ">", 0], ["-", 0, "-"], [0, ">", 0]], dtype=object)

        self.assertEqual(M.shape, (3, 3))
        self.assertEqual(len(node_index), 3)
        self.assertEqual(expected.tolist(), M.tolist())
        self.assertEqual(set(node_index.keys()), {"A", "B", "C"})

    def test_adjacency_matrix_empty_graph(self):
        """Test adjacency matrix for empty graph."""
        graph = AncestralBase()
        M, node_index = graph.adjacency_matrix
        self.assertEqual(M.shape, (0, 0))
        self.assertEqual(len(node_index), 0)

    def test_adjacency_matrix_setter(self):
        """Test setting graph from adjacency matrix."""
        M = np.array([[0, ">", 0], ["-", 0, ">"], [0, "-", 0]], dtype=object)
        graph = AncestralBase()
        graph.adjacency_matrix = M

        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 2)

        self.assertTrue(graph.has_edge("X_0", "X_1"))
        self.assertTrue(graph.has_edge("X_1", "X_2"))

        self.assertEqual(graph["X_0"]["X_1"]["marks"], {"X_0": ">", "X_1": "-"})
        self.assertEqual(graph["X_1"]["X_2"]["marks"], {"X_1": ">", "X_2": "-"})
