import unittest

import networkx as nx

from pgmpy.base.AncestralBase import AncestralBase


class TestAncestralBase(unittest.TestCase):

    def setUp(self):
        self.graph = AncestralBase()

    def test_init_empty(self):
        """Test initialization of empty graph."""
        graph = AncestralBase()
        self.assertIsInstance(graph.graph, nx.DiGraph)
        self.assertEqual(len(graph.graph.nodes), 0)
        self.assertEqual(len(graph.graph.edges), 0)

    def test_init_with_edges(self):
        edges = [("A", "B", "tail", "arrowhead"), ("B", "C", "tail", "arrowhead")]
        graph = AncestralBase(ebunch=edges)
        self.assertTrue(graph.graph.has_edge("A", "B"))
        self.assertTrue(graph.graph.has_edge("B", "C"))

    def test_add_edge_directed(self):
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.assertTrue(self.graph.graph.has_edge("A", "B"))
        self.assertTrue(self.graph.has_arrowhead_at("B", "A"))
        self.assertFalse(self.graph.has_arrowhead_at("A", "B"))

    def test_add_edge_bidirected(self):
        """Test adding a bidirected edge (arrowhead <-> arrowhead)."""
        self.graph.add_edge("A", "B", "arrowhead", "arrowhead")
        self.assertTrue(self.graph.graph.has_edge("A", "B"))
        self.assertTrue(self.graph.graph.has_edge("B", "A"))
        self.assertEqual(self.graph.graph.get_edge_data("A", "B")["mark"], "arrowhead")
        self.assertEqual(self.graph.graph.get_edge_data("B", "A")["mark"], "arrowhead")

    def test_add_edge_circle(self):
        """Test adding a circle edge (circle o-o circle)."""
        self.graph.add_edge("A", "B", "circle", "circle")
        self.assertTrue(self.graph.graph.has_edge("A", "B"))
        self.assertTrue(self.graph.graph.has_edge("B", "A"))
        self.assertEqual(self.graph.graph.get_edge_data("A", "B")["mark"], "circle")
        self.assertEqual(self.graph.graph.get_edge_data("B", "A")["mark"], "circle")

    def test_add_edge_invalid_marks(self):
        """Test that invalid marks raise ValueError."""
        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "B", "invalid", "arrowhead")

        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "B", "tail", "invalid")

        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "B", "invalid1", "invalid2")

    def test_add_edges_from(self):
        """Test adding multiple edges from a list."""
        edges = [
            ("A", "B", "tail", "arrowhead"),
            ("B", "C", "arrowhead", "arrowhead"),
            ("C", "D", "circle", "circle"),
        ]
        self.graph.add_edges_from(edges)

        self.assertTrue(self.graph.graph.has_edge("A", "B"))
        self.assertTrue(self.graph.graph.has_edge("B", "C"))
        self.assertTrue(self.graph.graph.has_edge("C", "B"))
        self.assertTrue(self.graph.graph.has_edge("C", "D"))
        self.assertTrue(self.graph.graph.has_edge("D", "C"))

    def test_is_directed_simple(self):
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.assertTrue(self.graph.is_directed("A", "B"))
        self.assertFalse(self.graph.is_directed("B", "A"))

    def test_is_directed_with_reverse_edge(self):
        """Test when both directions exist."""
        self.graph.graph.add_edge("A", "B", mark="tail")
        self.graph.graph.add_edge("B", "A", mark="arrowhead")
        self.assertTrue(self.graph.is_directed("A", "B"))
        self.assertFalse(self.graph.is_directed("B", "A"))

    def test_is_directed_false_cases(self):
        """Test cases where is_directed should return False."""
        # Bidirected edge
        self.graph.add_edge("A", "B", "arrowhead", "arrowhead")
        self.assertFalse(self.graph.is_directed("A", "B"))

        # Non-existent edge
        self.assertFalse(self.graph.is_directed("X", "Y"))

    def test_is_bidirected(self):
        """Test is_bidirected method."""
        # True case
        self.graph.add_edge("A", "B", "arrowhead", "arrowhead")
        self.assertTrue(self.graph.is_bidirected("A", "B"))
        self.assertTrue(self.graph.is_bidirected("B", "A"))

        # False cases
        self.graph.add_edge("C", "D", "tail", "arrowhead")
        self.assertFalse(self.graph.is_bidirected("C", "D"))

        # Non-existent edge
        self.assertFalse(self.graph.is_bidirected("X", "Y"))

    def test_has_arrowhead_at(self):
        """Test has_arrowhead_at method."""
        self.graph.add_edge("A", "B", "arrowhead", "tail")
        self.assertTrue(self.graph.has_arrowhead_at("A", "B"))
        self.assertFalse(self.graph.has_arrowhead_at("B", "A"))

        # Non-existent edge
        self.assertFalse(self.graph.has_arrowhead_at("X", "Y"))

    def test_has_circle_at(self):
        """Test has_circle_at method."""
        self.graph.add_edge("A", "B", "circle", "circle")
        self.assertTrue(self.graph.has_circle_at("A", "B"))
        self.assertTrue(self.graph.has_circle_at("B", "A"))

        # Non-circle edge
        self.graph.add_edge("C", "D", "tail", "arrowhead")
        self.assertFalse(self.graph.has_circle_at("C", "D"))

        # Non-existent edge
        self.assertFalse(self.graph.has_circle_at("X", "Y"))

    def test_has_tail_at(self):
        self.graph.graph.add_edge("A", "B", mark="arrowhead")
        self.graph.graph.add_edge("B", "A", mark="tail")

        self.assertTrue(self.graph.has_tail_at("B", "A"))
        self.assertFalse(self.graph.has_tail_at("A", "B"))

        # Non-existent edge
        self.assertFalse(self.graph.has_tail_at("X", "Y"))

    def test_get_parents(self):
        # A -> B -> C
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.graph.add_edge("B", "C", "tail", "arrowhead")

        self.assertEqual(self.graph.get_parents("B"), {"A"})
        self.assertEqual(self.graph.get_parents("C"), {"B"})
        self.assertEqual(self.graph.get_parents("A"), set())

        # Test with bidirected edge (should not be parent)
        self.graph.add_edge("D", "E", "arrowhead", "arrowhead")
        self.assertEqual(self.graph.get_parents("E"), set())

    def test_get_children(self):
        # A -> B -> C
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.graph.add_edge("B", "C", "tail", "arrowhead")

        self.assertEqual(self.graph.get_children("A"), {"B"})
        self.assertEqual(self.graph.get_children("B"), {"C"})
        self.assertEqual(self.graph.get_children("C"), set())

        # Test with bidirected edge (should not be child)
        self.graph.add_edge("D", "E", "arrowhead", "arrowhead")
        self.assertEqual(self.graph.get_children("D"), set())

    def test_get_spouses(self):
        # Bidirected edge A <-> B
        self.graph.add_edge("A", "B", "arrowhead", "arrowhead")
        self.assertEqual(self.graph.get_spouses("A"), {"B"})
        self.assertEqual(self.graph.get_spouses("B"), {"A"})

        # Directed edge should not create spouses
        self.graph.add_edge("C", "D", "tail", "arrowhead")
        self.assertEqual(self.graph.get_spouses("C"), set())
        self.assertEqual(self.graph.get_spouses("D"), set())

    def test_get_ancestors(self):
        # A -> B -> C -> D
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.graph.add_edge("B", "C", "tail", "arrowhead")
        self.graph.add_edge("C", "D", "tail", "arrowhead")

        self.assertEqual(self.graph.get_ancestors("D"), {"A", "B", "C"})
        self.assertEqual(self.graph.get_ancestors("C"), {"A", "B"})
        self.assertEqual(self.graph.get_ancestors("B"), {"A"})
        self.assertEqual(self.graph.get_ancestors("A"), set())

        # Test with multiple parents
        self.graph.add_edge("E", "D", "tail", "arrowhead")
        self.assertEqual(self.graph.get_ancestors("D"), {"A", "B", "C", "E"})

    def test_get_descendants(self):
        # A -> B -> C -> D
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.graph.add_edge("B", "C", "tail", "arrowhead")
        self.graph.add_edge("C", "D", "tail", "arrowhead")

        self.assertEqual(self.graph.get_descendants("A"), {"B", "C", "D"})
        self.assertEqual(self.graph.get_descendants("B"), {"C", "D"})
        self.assertEqual(self.graph.get_descendants("C"), {"D"})
        self.assertEqual(self.graph.get_descendants("D"), set())

        # Test with multiple children
        self.graph.add_edge("B", "E", "tail", "arrowhead")
        self.assertEqual(self.graph.get_descendants("B"), {"C", "D", "E"})

    def test_complex_graph_structure(self):
        """Test with a complex graph structure combining different edge types."""
        # Create a complex graph: A -> B <-> C o-o D -> E
        self.graph.add_edge("A", "B", "tail", "arrowhead")  # A -> B
        self.graph.add_edge("B", "C", "arrowhead", "arrowhead")  # B <-> C
        self.graph.add_edge("C", "D", "circle", "circle")  # C o-o D
        self.graph.add_edge("D", "E", "tail", "arrowhead")  # D -> E

        # Test various relationships
        self.assertEqual(self.graph.get_parents("B"), {"A"})
        self.assertEqual(self.graph.get_children("A"), {"B"})
        self.assertEqual(self.graph.get_spouses("B"), {"C"})
        self.assertEqual(self.graph.get_spouses("C"), {"B"})

        # Test directed relationships
        self.assertTrue(self.graph.is_directed("A", "B"))
        self.assertTrue(self.graph.is_directed("D", "E"))
        self.assertTrue(self.graph.is_bidirected("B", "C"))

        # Test mark detection
        self.assertTrue(self.graph.has_arrowhead_at("B", "A"))
        self.assertTrue(self.graph.has_tail_at("A", "B"))
        self.assertTrue(self.graph.has_circle_at("C", "D"))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with same node
        with self.assertRaises(ValueError):
            self.graph.add_edge("A", "A", "tail", "arrowhead")

        # Test methods on non-existent nodes
        self.assertEqual(self.graph.get_parents("NonExistent"), set())
        self.assertEqual(self.graph.get_children("NonExistent"), set())
        self.assertEqual(self.graph.get_spouses("NonExistent"), set())
        self.assertEqual(self.graph.get_ancestors("NonExistent"), set())
        self.assertEqual(self.graph.get_descendants("NonExistent"), set())

    def test_graph_modification_after_creation(self):
        """Test that the graph can be modified after creation."""
        # Start with empty graph
        self.assertEqual(len(self.graph.graph.nodes), 0)

        # Add edges incrementally
        self.graph.add_edge("A", "B", "tail", "arrowhead")
        self.assertEqual(len(self.graph.graph.nodes), 2)

        self.graph.add_edge("B", "C", "arrowhead", "arrowhead")
        self.assertEqual(len(self.graph.graph.nodes), 3)

        # Verify final structure
        self.assertTrue(self.graph.is_directed("A", "B"))
        self.assertTrue(self.graph.is_bidirected("B", "C"))
