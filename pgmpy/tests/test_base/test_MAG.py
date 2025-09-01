import unittest

from pgmpy.base.MAG import MAG


class TestMAG(unittest.TestCase):
    def setUp(self):

        edges = [
            ("X", "Y", "-", ">"),
            ("Y", "Z", "-", ">"),
            ("A", "B", ">", ">"),
            ("C", "L", "-", ">"),
            ("D", "L", "-", ">"),
            ("L", "E", "-", ">"),
            ("F", "G", "-", "-"),
            ("B", "H", ">", "-"),
            ("B", "I", "-", "-"),
        ]
        self.mag = MAG(ebunch=edges)

    def test_init_empty(self):
        """Test basic MAG initialization."""
        empty_mag = MAG()
        self.assertEqual(len(empty_mag.nodes()), 0)
        self.assertEqual(empty_mag.latents, set())

    def test_is_collider(self):
        """Test _is_collider for actual collider."""
        self.assertTrue(self.mag._is_collider("C", "L", "D"))
        self.assertFalse(self.mag._is_collider("X", "Y", "Z"))

    def test_is_collider_bidirected(self):
        """Test _is_collider with bidirected edges."""
        self.mag.add_edge("F", "A", "-", ">")
        self.assertTrue(self.mag._is_collider("F", "A", "B"))

    def test_has_inducing_path_true(self):
        """Test has_inducing_path when inducing path exists."""
        self.assertTrue(self.mag.has_inducing_path("C", "D", {"L"}))

    def test_has_inducing_path_false_no_path(self):
        """Test has_inducing_path when no path exists."""
        self.assertFalse(self.mag.has_inducing_path("X", "A", {"L"}))

    def test_has_inducing_path_false_direct_edge(self):
        """Test has_inducing_path returns False for direct connections."""
        self.assertFalse(self.mag.has_inducing_path("X", "Y", {"L"}))

    def test_is_visible_edge_true(self):
        """Test is_visible_edge for edges without latent confounders."""
        self.assertTrue(self.mag.is_visible_edge("X", "Y"))
        self.assertTrue(self.mag.is_visible_edge("A", "B"))
        self.assertTrue(self.mag.is_visible_edge("F", "G"))

    def test_is_visible_edge_false_no_edge(self):
        """Test is_visible_edge returns False for non-existent edges."""
        self.assertFalse(self.mag.is_visible_edge("X", "A"))
        self.assertFalse(self.mag.is_visible_edge("F", "Z"))

    def test_is_invisible_edge_true(self):
        """Test is_invisible_edge for edges with latent confounders."""
        self.mag.add_edge("C", "D", "-", ">")
        self.assertTrue(self.mag.is_invisible_edge("C", "D"))

    def test_is_invisible_edge_false(self):
        """Test is_invisible_edge for visible edges."""
        self.assertFalse(self.mag.is_invisible_edge("X", "Y"))
        self.assertFalse(self.mag.is_invisible_edge("A", "B"))

    def test_lower_manipulation_remove_latents(self):
        """Test lower_manipulation removes latent nodes."""
        new_mag = self.mag.lower_manipulation({"L"})
        self.assertNotIn("L", new_mag.nodes())
        self.assertIn("C", new_mag.nodes())
        self.assertIn("D", new_mag.nodes())
        self.assertIn("E", new_mag.nodes())

    def test_lower_manipulation_preserve_structure(self):
        """Test lower_manipulation preserves non-marginalized structure."""
        new_mag = self.mag.lower_manipulation({"L"})

        self.assertTrue(new_mag.has_edge("X", "Y"))
        self.assertTrue(new_mag.has_edge("Y", "Z"))
        self.assertTrue(new_mag.has_edge("A", "B"))
        self.assertTrue(new_mag.has_edge("F", "G"))

    def test_lower_manipulation_empty_set(self):
        """Test lower_manipulation with empty set."""
        new_mag = self.mag.lower_manipulation(set())

        self.assertEqual(set(new_mag.nodes()), set(self.mag.nodes()))
        self.assertEqual(len(new_mag.edges()), len(self.mag.edges()))

    def test_upper_manipulation_remove_outgoing(self):
        """Test upper_manipulation removes outgoing directed edges."""
        new_mag = self.mag.upper_manipulation({"X"})

        self.assertFalse(new_mag.has_edge("X", "Y"))

        self.assertTrue(new_mag.has_edge("Y", "Z"))
        self.assertTrue(new_mag.has_edge("A", "B"))

    def test_upper_manipulation_preserve_incoming(self):
        """Test upper_manipulation preserves incoming edges."""
        new_mag = self.mag.upper_manipulation({"Y"})

        self.assertTrue(new_mag.has_edge("X", "Y"))

        self.assertFalse(new_mag.has_edge("Y", "Z"))

    def test_upper_manipulation_bidirected_preserved(self):
        """Test upper_manipulation preserves bidirected edges."""
        new_mag = self.mag.upper_manipulation({"A"})

        self.assertTrue(new_mag.has_edge("A", "B"))

    def test_upper_manipulation_undirected_preserved(self):
        """Test upper_manipulation preserves undirected edges."""
        new_mag = self.mag.upper_manipulation({"F"})

        self.assertTrue(new_mag.has_edge("F", "G"))

    def test_upper_manipulation_empty_set(self):
        """Test upper_manipulation with empty set."""
        new_mag = self.mag.upper_manipulation(set())

        self.assertEqual(set(new_mag.nodes()), set(self.mag.nodes()))
        self.assertEqual(len(new_mag.edges()), len(self.mag.edges()))

    def test_manipulation_methods_dont_modify_original(self):
        """Test that manipulation methods don't modify the original graph."""
        original_nodes = set(self.mag.nodes())
        original_edges = len(self.mag.edges())

        self.mag.lower_manipulation({"L"})
        self.mag.upper_manipulation({"X"})

        self.assertEqual(set(self.mag.nodes()), original_nodes)
        self.assertEqual(len(self.mag.edges()), original_edges)

    def test_edge_visibility_mutual_exclusivity(self):
        """Test that existing edges are either visible or invisible, not both."""
        for u, v in self.mag.edges():
            is_visible = self.mag.is_visible_edge(u, v)
            is_invisible = self.mag.is_invisible_edge(u, v)

            self.assertTrue(
                is_visible != is_invisible,
                f"Edge ({u}, {v}) should be either visible or invisible",
            )

    def test_nonexistent_edge_cases(self):
        """Test behavior with non-existent edges."""
        self.assertFalse(self.mag.is_visible_edge("X", "A"))
        self.assertFalse(self.mag.is_invisible_edge("X", "A"))

    def test_collider_edge_cases(self):
        """Test _is_collider with edge cases."""
        with self.assertRaises(KeyError):
            self.mag._is_collider("X", "NONEXISTENT", "Y")

    def test_inducing_path_with_invisible_edge(self):
        """Test complete scenario: invisible edge due to inducing path."""
        self.mag.add_edge("C", "D", "-", ">")

        self.assertTrue(self.mag.is_invisible_edge("C", "D"))
        self.assertFalse(self.mag.is_visible_edge("C", "D"))

    def test_complex_graph_properties(self):
        """Test properties of the complex graph setup."""
        self.assertEqual(len(self.mag.nodes()), 11)
        self.assertIn("L", self.mag.latents)

        self.assertTrue(self.mag.has_edge("X", "Y"))
        self.assertTrue(self.mag.has_edge("C", "L"))
        self.assertTrue(self.mag.has_edge("D", "L"))
        self.assertTrue(self.mag.has_edge("L", "E"))
