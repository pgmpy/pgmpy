from itertools import permutations

from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.example_models import load_model


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

    def test_generate_screening_search_space(self):
        model = load_model("bnlearn/cancer")
        data = model.simulate(n_samples=5000, seed=42)

        ek = ExpertKnowledge(
            screening_method="chi_square",
            significance_level=0.05,
        )

        ek._generate_screening_search_space(data)

        assert len(ek.search_space) > 0

    def test_limit_search_space_with_screening(self):
        model = load_model("bnlearn/cancer")
        data = model.simulate(n_samples=5000, seed=42)

        ek = ExpertKnowledge(
            screening_method="chi_square",
            significance_level=0.05,
        )

        ek.limit_search_space(data)
        all_possible_edges = set(permutations(data.columns, 2))
        expected_forbidden = all_possible_edges - ek.search_space

        assert ek.forbidden_edges == expected_forbidden
