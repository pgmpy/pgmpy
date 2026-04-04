import pytest

from pgmpy.causal_discovery import ExpertKnowledge


class TestExpertKnowledge:
    def test_repr_and_str_empty(self):
        ek = ExpertKnowledge()
        assert repr(ek) == (
            "Expert Knowledge: 0 required edges, 0 forbidden edges, "
            "temporal order on 0 nodes, 0 search space edges, and 0 explicit orientations"
        )
        assert str(ek) == "Expert Knowledge:"

    def test_repr_and_str_populated(self):
        ek = ExpertKnowledge(
            required_edges=[("A", "B")],
            temporal_order=[["A"], ["B"]],
            forbidden_edges=[("C", "D")],
            search_space=[("A", "B"), ("B", "C")],
            orientations=[("A", "B")],
        )
        # Call str and repr directly to ensure coverage
        s = str(ek)
        r = repr(ek)
        print(s)
        print(r)

        assert repr(ek) == (
            "Expert Knowledge: 1 required edges, 1 forbidden edges, "
            "temporal order on 2 nodes, 2 search space edges, and 1 explicit orientations"
        )
        assert "Expert Knowledge:\n" in str(ek)
        assert "Required Edges: {('A', 'B')}" in str(ek)
        assert "Forbidden Edges: {('C', 'D')}" in str(ek)
        assert "Search Space: {" in str(ek)  # Sets are unordered, so check prefix and then individual elements.

        # Check individual elements to avoid flakiness with set representation
        assert "('A', 'B')" in str(ek)
        assert "('B', 'C')" in str(ek)
        assert "Orientations: {('A', 'B')}" in str(ek)
        assert "Temporal Order: [['A'], ['B']]" in str(ek)

    def test_duplicate_node_in_temporal_order_raises(self):
        """ValueError is raised when a node appears in multiple tiers of temporal_order.

        Covers ExpertKnowledge._get_temporal_ordering line 192.
        """
        with pytest.raises(ValueError, match="present in multiple tiers"):
            ExpertKnowledge(temporal_order=[["A", "B"], ["A", "C"]])

    def test_validate_edges_type_error(self):
        """Line 198 in ExpertKnowledge.py: invalid edge_list type."""
        ek = ExpertKnowledge()
        with pytest.raises(TypeError, match="edge_list must be a list, tuple, or set"):
            ek._validate_edges("invalid")

    def test_validate_edges_rejects_bad_formats(self):
        with pytest.raises(TypeError):
            ExpertKnowledge(forbidden_edges="A->B")
        with pytest.raises(ValueError):
            ExpertKnowledge(forbidden_edges=[("A", "B", "C")])
        # Accepts list/tuple, coerces to set of tuples
        ek = ExpertKnowledge(required_edges=[["A", "B"], ("C", "D")])
        assert ("A", "B") in ek.required_edges and ("C", "D") in ek.required_edges

    def test_temporal_ordering_duplicate_raises(self):
        with pytest.raises(ValueError, match="present in multiple tiers"):
            ExpertKnowledge(temporal_order=[["A"], ["A"]])._get_temporal_ordering([["A"], ["A"]])

    def test_validate_temporal_order_missing_nodes_raises(self):
        ek = ExpertKnowledge(temporal_order=[["A"], ["B"]])
        with pytest.raises(ValueError, match="Missing nodes in temporal order"):
            ek._validate_temporal_order(nodes=["A", "B", "C"])

    def test_limit_search_space_adds_forbidden(self):
        labels = {"A", "B", "C"}
        # search_space allows only A->B
        ek = ExpertKnowledge(search_space=[("A", "B")])
        ek.limit_search_space(labels)
        # Then all other directed pairs become forbidden
        expected_forbidden = {("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")}
        assert expected_forbidden.issubset(ek.forbidden_edges)
