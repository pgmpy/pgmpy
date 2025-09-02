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
        latents = {"L"}
        self.mag = MAG(ebunch=edges, latents=latents)

    def test_init_empty(self):
        empty_mag = MAG()
        self.assertEqual(len(empty_mag.nodes()), 0)
        self.assertEqual(empty_mag.latents, set())

    def test_is_collider(self):
        self.assertTrue(self.mag._is_collider("C", "L", "D"))
        self.assertFalse(self.mag._is_collider("X", "Y", "Z"))

    def test_has_inducing_path(self):
        self.assertTrue(self.mag.has_inducing_path("C", "D", {"L"}))
        self.assertFalse(self.mag.has_inducing_path("X", "A", {"L"}))
        self.assertFalse(self.mag.has_inducing_path("X", "Y", {"L"}))

    def test_edge_visibility(self):
        self.assertTrue(self.mag.is_visible_edge("X", "Y"))
        self.assertTrue(self.mag.is_visible_edge("A", "B"))
        self.assertTrue(self.mag.is_visible_edge("F", "G"))
        self.mag.add_edge("C", "D", "-", ">")
        self.assertTrue(self.mag.is_invisible_edge("C", "D"))
        self.assertFalse(self.mag.is_visible_edge("C", "D"))

    def test_edge_visibility_mutual_exclusivity(self):
        for u, v in self.mag.edges():
            is_visible = self.mag.is_visible_edge(u, v)
            is_invisible = self.mag.is_invisible_edge(u, v)
            self.assertNotEqual(is_visible, is_invisible)

    def test_lower_manipulation(self):
        new_mag = self.mag.lower_manipulation({"L"})
        self.assertNotIn("L", new_mag.nodes())
        self.assertTrue(new_mag.has_edge("X", "Y"))
        self.assertTrue(new_mag.has_edge("F", "G"))

    def test_upper_manipulation(self):
        new_mag = self.mag.upper_manipulation({"X"})
        self.assertFalse(new_mag.has_edge("X", "Y"))
        self.assertTrue(new_mag.has_edge("Y", "Z"))

        new_mag = self.mag.upper_manipulation({"Y"})
        self.assertTrue(new_mag.has_edge("X", "Y"))
        self.assertFalse(new_mag.has_edge("Y", "Z"))

        new_mag = self.mag.upper_manipulation({"B"})
        self.assertTrue(new_mag.has_edge("A", "B"))
        self.assertTrue(new_mag.has_edge("B", "I"))

    def test_manipulations_do_not_modify_original(self):
        original_nodes = set(self.mag.nodes())
        original_edges = set(self.mag.edges())
        self.mag.lower_manipulation({"L"})
        self.mag.upper_manipulation({"X"})
        self.assertEqual(set(self.mag.nodes()), original_nodes)
        self.assertEqual(set(self.mag.edges()), original_edges)

    def test_graph_properties(self):
        self.assertEqual(len(self.mag.nodes()), 13)
        self.assertIn("L", self.mag.latents)
        self.assertTrue(self.mag.has_edge("B", "H"))
        self.assertTrue(self.mag.has_edge("B", "I"))
