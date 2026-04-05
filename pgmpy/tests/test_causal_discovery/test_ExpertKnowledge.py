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
        assert repr(ek) == (
            "Expert Knowledge: 1 required edges, 1 forbidden edges, "
            "temporal order on 2 nodes, 2 search space edges, and 1 explicit orientations"
        )
        assert "Expert Knowledge:\n" in str(ek)
        assert "Required Edges: {('A', 'B')}" in str(ek)
        assert "Forbidden Edges: {('C', 'D')}" in str(ek)
        assert "Search Space: {" in str(ek)
        assert "('A', 'B')" in str(ek)
        assert "('B', 'C')" in str(ek)
        assert "Orientations: {('A', 'B')}" in str(ek)
        assert "Temporal Order: [['A'], ['B']]" in str(ek)

    def test_duplicate_node_in_temporal_order_raises(self):
        with pytest.raises(ValueError, match="present in multiple tiers"):
            ExpertKnowledge(temporal_order=[["A", "B"], ["A", "C"]])

    def test_validate_edges_type_error(self):
        ek = ExpertKnowledge()
        with pytest.raises(TypeError, match="edge_list must be iterable"):
            ek._validate_edges(12345)

    def test_validate_edges_rejects_bad_formats(self):
        ek1 = ExpertKnowledge(forbidden_edges=["A->B"])
        assert ("A", "B") in ek1.forbidden_edges
        
        ek2 = ExpertKnowledge(forbidden_edges=["C-D"])
        assert ("C", "D") in ek2.forbidden_edges
        
        with pytest.raises(ValueError, match="Invalid edge format.*Expected.*pair"):
            ExpertKnowledge(forbidden_edges=[("A", "B", "C")])
        
        with pytest.raises(ValueError, match="Invalid edge format: A"):
            ExpertKnowledge(forbidden_edges=["A"])
        
        with pytest.raises(TypeError, match="iterable"):
            ExpertKnowledge(forbidden_edges=12345)
        
        ek3 = ExpertKnowledge(required_edges=[["A", "B"], ("C", "D")])
        assert ("A", "B") in ek3.required_edges and ("C", "D") in ek3.required_edges

    def test_temporal_ordering_duplicate_raises(self):
        with pytest.raises(ValueError, match="present in multiple tiers"):
            ExpertKnowledge(temporal_order=[["A"], ["A"]])

    def test_validate_temporal_order_missing_nodes_raises(self):
        ek = ExpertKnowledge(temporal_order=[["A"], ["B"]])
        with pytest.raises(ValueError, match="Missing nodes in temporal order"):
            ek._validate_temporal_order(nodes=["A", "B", "C"])

    def test_limit_search_space_adds_forbidden(self):
        labels = {"A", "B", "C"}
        ek = ExpertKnowledge(search_space=[("A", "B")])
        ek.limit_search_space(labels)
        expected_forbidden = {("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")}
        assert expected_forbidden.issubset(ek.forbidden_edges)

    def test_validate_edges_string_formats(self):
        ek1 = ExpertKnowledge(required_edges=["A->B", "C->D"])
        assert ("A", "B") in ek1.required_edges
        assert ("C", "D") in ek1.required_edges
        
        ek2 = ExpertKnowledge(forbidden_edges=["E-F", "G-H"])
        assert ("E", "F") in ek2.forbidden_edges
        assert ("G", "H") in ek2.forbidden_edges
        
        ek3 = ExpertKnowledge(search_space=["X->Y", "Z-W"])
        assert ("X", "Y") in ek3.search_space
        assert ("Z", "W") in ek3.search_space
        ek4 = ExpertKnowledge(orientations=["A -> B", "C - D"])
        assert ("A", "B") in ek4.orientations
        assert ("C", "D") in ek4.orientations

    def test_validate_edges_invalid_strings(self):
        with pytest.raises(ValueError, match="Invalid edge format: A=>B"):
            ExpertKnowledge(required_edges=["A=>B"])
        with pytest.raises(ValueError, match="Invalid edge format: empty string"):
            ExpertKnowledge(required_edges=[""])
        with pytest.raises(ValueError, match="exactly one '->'"):
            ExpertKnowledge(required_edges=["A->B->C"])
        with pytest.raises(ValueError, match="Expected 'u->v' or 'u-v' string"):
            ExpertKnowledge(required_edges=["X"])
        with pytest.raises(ValueError, match="exactly one '-'"):
            ExpertKnowledge(required_edges=["A-B-C"])

    def test_validate_edges_accepts_generators(self):
        gen = (("A", "B") for _ in range(1))
        ek1 = ExpertKnowledge(required_edges=gen)
        assert ("A", "B") in ek1.required_edges
        
        ek2 = ExpertKnowledge(forbidden_edges=map(lambda x: (x, x+"1"), ["A", "B"]))
        assert ("A", "A1") in ek2.forbidden_edges
        assert ("B", "B1") in ek2.forbidden_edges
        class EdgeIterable:
            def __iter__(self):
                return iter([("X", "Y"), ("Y", "Z")])
        
        ek3 = ExpertKnowledge(search_space=EdgeIterable())
        assert ("X", "Y") in ek3.search_space
        assert ("Y", "Z") in ek3.search_space