from pgmpy.causal_discovery import ExpertKnowledge


class TestExpertKnowledge:
    def test_repr_and_str_empty(self):
        ek = ExpertKnowledge()
        assert repr(ek) == (
            "Expert Knowledge: 0 required edges, 0 forbidden edges, temporal order on 0 nodes, and 0 search space edges"
        )
        assert str(ek) == "Expert Knowledge:"

    def test_repr_and_str_populated(self):
        ek = ExpertKnowledge(
            required_edges=[("A", "B")],
            temporal_order=[["A"], ["B"]],
            forbidden_edges=[("C", "D")],
            search_space=[("A", "B"), ("B", "C")],
        )
        assert repr(ek) == (
            "Expert Knowledge: 1 required edges, 1 forbidden edges, temporal order on 2 nodes, and 2 search space edges"
        )
        assert "Expert Knowledge:\n" in str(ek)
        assert "Required Edges: {('A', 'B')}" in str(ek)
        assert "Forbidden Edges: {('C', 'D')}" in str(ek)
        assert "Search Space: {" in str(ek)  # Sets are unordered, so check prefix and then individual elements.

        # Check individual elements to avoid flakiness with set representation
        assert "('A', 'B')" in str(ek)
        assert "('B', 'C')" in str(ek)
        assert "Temporal Order: [['A'], ['B']]" in str(ek)
