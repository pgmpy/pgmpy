import unittest

from pgmpy.causal_discovery import ExpertKnowledge


class TestExpertKnowledge(unittest.TestCase):
    def test_repr_and_str_empty(self):
        ek = ExpertKnowledge()
        self.assertEqual(
            repr(ek),
            "<ExpertKnowledge with 0 required edges, 0 forbidden edges, 0 temporal tiers, and 0 search space edges>",
        )
        self.assertIn("Required Edges: None", str(ek))
        self.assertIn("Forbidden Edges: None", str(ek))
        self.assertIn("Search Space: None", str(ek))
        self.assertIn("Temporal Order: None", str(ek))

    def test_repr_and_str_populated(self):
        ek = ExpertKnowledge(
            required_edges=[("A", "B")],
            temporal_order=[["A"], ["B"]],
            forbidden_edges=[("C", "D")],
            search_space=[("A", "B"), ("B", "C")],
        )
        self.assertEqual(
            repr(ek),
            "<ExpertKnowledge with 1 required edges, 1 forbidden edges, 2 temporal tiers, and 2 search space edges>",
        )
        self.assertIn("Required Edges: {('A', 'B')}", str(ek))
        self.assertIn("Forbidden Edges: {('C', 'D')}", str(ek))
        self.assertIn("Search Space: {", str(ek))  # Sets are unordered, so check prefix and then individual elements.

        # Check individual elements to avoid flakiness with set representation
        self.assertIn("('A', 'B')", str(ek))
        self.assertIn("('B', 'C')", str(ek))
        self.assertIn("Temporal Order: [['A'], ['B']]", str(ek))
