from itertools import permutations

import pytest

from pgmpy.estimators import ExpertKnowledge
from pgmpy.example_models import load_model


class TestExpertKnowledgeDeprecated:
    def test_emits_deprecation_warning(self):
        with pytest.warns(FutureWarning, match="ExpertKnowledge is deprecated"):
            ExpertKnowledge(forbidden_edges=[("A", "B")])

    def test_generate_screening_search_space(self):
        model = load_model("bnlearn/cancer")
        data = model.simulate(n_samples=5000, seed=42)

        with pytest.warns(FutureWarning):
            ek = ExpertKnowledge(screening_method="chi_square", significance_level=0.05)

        ek._generate_screening_search_space(data)

        assert len(ek.search_space) > 0

    def test_limit_search_space_with_screening(self):
        model = load_model("bnlearn/cancer")
        data = model.simulate(n_samples=5000, seed=42)

        with pytest.warns(FutureWarning):
            ek = ExpertKnowledge(screening_method="chi_square", significance_level=0.05)

        ek.limit_search_space(data)
        all_possible_edges = set(permutations(data.columns, 2))
        expected_forbidden = all_possible_edges - ek.search_space

        assert ek.forbidden_edges == expected_forbidden
