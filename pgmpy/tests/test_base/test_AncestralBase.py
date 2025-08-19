import unittest

import networkx as nx

from pgmpy.base.AncestralBase import AncestralBase


class TestAncestralBase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.graph = AncestralBase()

    def test_init_empty(self):
        """Test initialization of empty graph."""
        graph = AncestralBase()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)

    def test_init_with_edges(self):
        """Test initialization with edge list."""
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        graph = AncestralBase(edges)
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 2)
        self.assertIn("A", graph.nodes)
        self.assertIn("B", graph.nodes)
        self.assertIn("C", graph.nodes)

    def test_add_edge_valid(self):
        """Test adding valid edges."""
        self.graph.add_edge("A", "B", "-", ">")
        self.assertTrue(self.graph.has_edge("A", "B"))
        self.assertEqual(self.graph.get_edge_data("A", "B")["marks"], ("-", ">"))

    def test_add_edge_invalid_same_node(self):
        """Test that adding edge with same source and target raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.graph.add_edge("A", "A", "-", ">")
        self.assertIn("Nodes cannot be the same", str(cm.exception))

    def test_add_edge_invalid_marks(self):
        """Test that invalid marks raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.graph.add_edge("A", "B", "x", ">")
        self.assertIn("Marks must be one of", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.graph.add_edge("A", "B", "-", "y")
        self.assertIn("Marks must be one of", str(cm.exception))

    def test_add_edge_valid_marks(self):
        """Test all valid mark combinations."""
        valid_marks = ["-", ">", "o"]
        for u_mark in valid_marks:
            for v_mark in valid_marks:
                graph = AncestralBase()
                graph.add_edge("A", "B", u_mark, v_mark)
                self.assertEqual(
                    graph.get_edge_data("A", "B")["marks"], (u_mark, v_mark)
                )

    def test_add_edges_from(self):
        """Test adding multiple edges at once."""
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "o"), ("C", "D", "o", "-")]
        self.graph.add_edges_from(edges)

        self.assertEqual(len(self.graph.edges), 3)
        self.assertEqual(self.graph.get_edge_data("A", "B")["marks"], ("-", ">"))
        self.assertEqual(self.graph.get_edge_data("B", "C")["marks"], (">", "o"))
        self.assertEqual(self.graph.get_edge_data("C", "D")["marks"], ("o", "-"))

    def test_get_marks(self):
        """Test _get_marks method for both edge directions."""
        self.graph.add_edge("A", "B", "-", ">")

        # Direct edge A -> B
        marks = self.graph._get_marks("A", "B")
        self.assertEqual(marks, ("-", ">"))

        # Reverse direction B -> A should reverse marks
        marks = self.graph._get_marks("B", "A")
        self.assertEqual(marks, (">", "-"))

    def test_to_adjacency_matrix_empty(self):
        """Test adjacency matrix for empty graph."""
        M, node_index = self.graph.to_adjacency_matrix()
        self.assertEqual(M.shape, (0, 0))
        self.assertEqual(node_index, {})

    def test_to_adjacency_matrix_single_edge(self):
        """Test adjacency matrix with single edge."""
        self.graph.add_edge("A", "B", "-", ">")
        M, node_index = self.graph.to_adjacency_matrix()

        self.assertEqual(M.shape, (2, 2))
        self.assertEqual(set(node_index.keys()), {"A", "B"})

        a_idx = node_index["A"]
        b_idx = node_index["B"]

        # A -> B has mark '>' at B
        self.assertEqual(M[a_idx, b_idx], ">")
        # B -> A has mark '-' at A
        self.assertEqual(M[b_idx, a_idx], "-")
        # Diagonal should be empty
        self.assertEqual(M[a_idx, a_idx], "")
        self.assertEqual(M[b_idx, b_idx], "")

    def test_to_adjacency_matrix_multiple_edges(self):
        """Test adjacency matrix with multiple edges."""
        edges = [("A", "B", "-", ">"), ("B", "C", ">", "o"), ("A", "C", "o", "-")]
        self.graph.add_edges_from(edges)
        M, node_index = self.graph.to_adjacency_matrix()

        self.assertEqual(M.shape, (3, 3))
        self.assertEqual(set(node_index.keys()), {"A", "B", "C"})

        a_idx = node_index["A"]
        b_idx = node_index["B"]
        c_idx = node_index["C"]

        # Check all edge marks
        self.assertEqual(M[a_idx, b_idx], ">")  # A -> B
        self.assertEqual(M[b_idx, a_idx], "-")  # B -> A
        self.assertEqual(M[b_idx, c_idx], "o")  # B -> C
        self.assertEqual(M[c_idx, b_idx], ">")  # C -> B
        self.assertEqual(M[a_idx, c_idx], "-")  # A -> C
        self.assertEqual(M[c_idx, a_idx], "o")  # C -> A

    def test_get_neighbors_no_constraints(self):
        """Test getting all neighbors without constraints."""
        self.graph.add_edges_from([("A", "B", "-", ">"), ("A", "C", ">", "o")])
        neighbors = self.graph.get_neighbors("A")
        self.assertEqual(neighbors, {"B", "C"})

    def test_get_neighbors_with_constraints(self):
        """Test getting neighbors with mark constraints."""
        self.graph.add_edges_from(
            [
                ("A", "B", "-", ">"),  # A-B>
                ("A", "C", ">", "o"),  # A>Co
                ("A", "D", "o", "-"),  # AoD-
            ]
        )

        # Neighbors where neighbor has '>' mark towards A
        neighbors = self.graph.get_neighbors("A", u_type=">")
        self.assertEqual(neighbors, {"C"})

        # Neighbors where A has '>' mark towards neighbor
        neighbors = self.graph.get_neighbors("A", v_type=">")
        self.assertEqual(neighbors, {"B"})

        # Both constraints
        neighbors = self.graph.get_neighbors("A", u_type=">", v_type="o")
        self.assertEqual(neighbors, {"C"})

    def test_get_neighbors_nonexistent_node(self):
        """Test getting neighbors of non-existent node."""
        neighbors = self.graph.get_neighbors("X")
        self.assertEqual(neighbors, set())

    def test_get_parents(self):
        """Test getting parent nodes (nodes with '>' pointing to target)."""
        self.graph.add_edges_from(
            [
                ("A", "C", "-", ">"),  # A-C> (A is parent of C)
                ("B", "C", ">", ">"),  # B>C> (B is parent of C)
                ("C", "D", ">", "-"),  # C>D- (C is not parent of D)
            ]
        )

        parents = self.graph.get_parents("C")
        self.assertEqual(parents, {"A", "B"})

        parents = self.graph.get_parents("D")
        self.assertEqual(parents, set())

    def test_get_children(self):
        """Test getting child nodes (nodes with '>' pointing from source)."""
        self.graph.add_edges_from(
            [
                ("A", "B", "-", ">"),  # A-B> (B is child of A)
                ("A", "C", ">", ">"),  # A>C> (C is child of A)
                ("D", "A", ">", "-"),  # D>A- (A is not child of D)
            ]
        )

        children = self.graph.get_children("A")
        self.assertEqual(children, {"B", "C"})

        children = self.graph.get_children("D")
        self.assertEqual(children, set())

    def test_get_spouses(self):
        """Test getting spouse nodes (bidirectional '>' marks)."""
        self.graph.add_edges_from(
            [
                ("A", "B", ">", ">"),  # A>B> (spouses)
                ("A", "C", "-", ">"),  # A-C> (not spouses)
                ("A", "D", ">", "o"),  # A>Do (not spouses)
            ]
        )

        spouses = self.graph.get_spouses("A")
        self.assertEqual(spouses, {"B"})

        spouses = self.graph.get_spouses("B")
        self.assertEqual(spouses, {"A"})

    def test_get_ancestors(self):
        """Test getting all ancestor nodes."""
        # Create a chain: E -> D -> C -> A, F -> C
        self.graph.add_edges_from(
            [
                ("A", "B", "-", ">"),  # A is parent of B
                ("C", "A", "-", ">"),  # C is parent of A
                ("D", "C", "-", ">"),  # D is parent of C
                ("E", "D", "-", ">"),  # E is parent of D
                ("F", "C", "-", ">"),  # F is parent of C
            ]
        )

        ancestors = self.graph.get_ancestors("B")
        self.assertEqual(ancestors, {"A", "C", "D", "E", "F"})

        ancestors = self.graph.get_ancestors("A")
        self.assertEqual(ancestors, {"C", "D", "E", "F"})

        ancestors = self.graph.get_ancestors("E")
        self.assertEqual(ancestors, set())

    def test_get_descendants(self):
        """Test getting all descendant nodes."""
        # Create a chain: A -> B -> C -> D, A -> E -> D
        self.graph.add_edges_from(
            [
                ("A", "B", "-", ">"),  # A is parent of B
                ("B", "C", "-", ">"),  # B is parent of C
                ("C", "D", "-", ">"),  # C is parent of D
                ("A", "E", "-", ">"),  # A is parent of E
                ("E", "D", "-", ">"),  # E is parent of D
            ]
        )

        descendants = self.graph.get_descendants("A")
        self.assertEqual(descendants, {"B", "C", "D", "E"})

        descendants = self.graph.get_descendants("B")
        self.assertEqual(descendants, {"C", "D"})

        descendants = self.graph.get_descendants("D")
        self.assertEqual(descendants, set())

    def test_get_ancestors_with_cycles(self):
        """Test ancestor detection doesn't get stuck in cycles."""
        # This shouldn't happen in a proper ancestral graph, but test robustness
        self.graph.add_edges_from(
            [
                ("A", "B", "-", ">"),
                ("B", "C", "-", ">"),
                ("C", "A", "-", ">"),  # Creates a cycle
            ]
        )

        ancestors = self.graph.get_ancestors("A")
        self.assertEqual(ancestors, {"B", "C"})  # Should handle cycle gracefully

    def test_get_descendants_with_cycles(self):
        """Test descendant detection doesn't get stuck in cycles."""
        self.graph.add_edges_from(
            [
                ("A", "B", "-", ">"),
                ("B", "C", "-", ">"),
                ("C", "A", "-", ">"),  # Creates a cycle
            ]
        )

        descendants = self.graph.get_descendants("A")
        self.assertEqual(descendants, {"B", "C"})  # Should handle cycle gracefully

    def test_inheritance_from_networkx(self):
        """Test that AncestralBase properly inherits from nx.DiGraph."""
        self.assertIsInstance(self.graph, nx.DiGraph)

        # Test that basic NetworkX functionality works
        self.graph.add_edge("A", "B", "-", ">")
        self.assertTrue(self.graph.has_node("A"))
        self.assertTrue(self.graph.has_node("B"))
        self.assertEqual(self.graph.number_of_nodes(), 2)
        self.assertEqual(self.graph.number_of_edges(), 1)
