import logging

import pandas as pd
import pytest

from pgmpy.base import PDAG
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

    def test_fit_temporal_required_forbidden(self):
        data = pd.DataFrame({c: [0, 1] for c in ["A", "B", "C", "D"]})
        ek = ExpertKnowledge(
            forbidden_edges=[("A", "B")],
            required_edges=[("C", "D")],
            temporal_order=[["A"], ["B", "C"], ["D"]],
        )
        ek.fit(data)

        assert ek.temporal_ordering_ == {"A": 0, "B": 1, "C": 1, "D": 2}
        assert ek.required_edges_ == {("C", "D")}
        # forbidden_edges_ = user forbidden + temporal complement (later tier -> earlier tier)
        assert ek.forbidden_edges_ == {
            ("A", "B"),  # user-specified
            ("B", "A"),
            ("C", "A"),  # tier 1 -> tier 0
            ("D", "A"),
            ("D", "B"),
            ("D", "C"),  # tier 2 -> lower tiers
        }
        # Pristine constructor attributes are untouched; fit is idempotent.
        assert ek.forbidden_edges == {("A", "B")}
        forbidden_first = set(ek.forbidden_edges_)
        ek.fit(data)
        assert ek.forbidden_edges_ == forbidden_first

    def test_fit_search_space_complement(self):
        data = pd.DataFrame({c: [0, 1] for c in ["A", "B", "C"]})
        ek = ExpertKnowledge(search_space=[("A", "B"), ("B", "C")])
        ek.fit(data)

        assert ek.search_space_ == {("A", "B"), ("B", "C")}
        # forbidden_edges_ = all directed pairs not in the search space
        assert ek.forbidden_edges_ == {("A", "C"), ("B", "A"), ("C", "A"), ("C", "B")}
        # pristine search_space untouched
        assert ek.search_space == {("A", "B"), ("B", "C")}

    def test_apply_to_orients_required_forbidden_and_warns(self, caplog):
        data = pd.DataFrame({c: [0, 1] for c in ["A", "B", "C", "D", "E", "F"]})
        ek = ExpertKnowledge(
            required_edges=[("A", "B"), ("E", "F")],
            forbidden_edges=[("C", "D")],
        )
        ek.fit(data)

        pdag = PDAG(
            directed_ebunch=[("F", "E")],
            undirected_ebunch=[("A", "B"), ("C", "D")],
        )

        pgmpy_logger = logging.getLogger("pgmpy")
        pgmpy_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING", logger="pgmpy"):
                ek.apply_to(pdag)
        finally:
            pgmpy_logger.removeHandler(caplog.handler)

        # required A-B oriented A->B; forbidden C->D oriented away (D->C)
        assert ("A", "B") in pdag.directed_edges
        assert ("D", "C") in pdag.directed_edges
        # required E->F conflicts with existing F->E: warned and left as-is
        assert ("F", "E") in pdag.directed_edges
        assert any("Ignoring edge E->F from required edges" in m for m in caplog.messages)

    def test_fit_without_data_resolves_declarative_knowledge(self):
        # forbidden/required/temporal knowledge needs no data, so fit() works without it.
        ek = ExpertKnowledge(
            forbidden_edges=[("A", "B")],
            required_edges=[("C", "D")],
            temporal_order=[["A"], ["B", "C"], ["D"]],
        )
        ek.fit()  # no data

        assert ek.temporal_ordering_ == {"A": 0, "B": 1, "C": 1, "D": 2}
        assert ek.required_edges_ == {("C", "D")}
        assert ek.forbidden_edges_ == {("A", "B"), ("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("D", "C")}
        assert ek.search_space_ == set()

    def test_fit_without_data_raises_when_data_required(self):
        # Screening and search-space resolution both need the dataset.
        with pytest.raises(ValueError, match="data"):
            ExpertKnowledge(screening_method="chi_square").fit()
        with pytest.raises(ValueError, match="data"):
            ExpertKnowledge(search_space=[("A", "B")]).fit()

    def test_is_sklearn_estimator(self):
        from sklearn.base import BaseEstimator, clone

        ek = ExpertKnowledge(required_edges=[("A", "B")], significance_level=0.1)
        assert isinstance(ek, BaseEstimator)
        # get_params exposes the constructor params (enables clone / nested params / grid search).
        assert ek.get_params()["significance_level"] == 0.1
        # clone produces an equivalent, unfitted copy.
        ek_clone = clone(ek)
        assert ek_clone is not ek
        assert ek_clone.required_edges == {("A", "B")}
        assert not hasattr(ek_clone, "required_edges_")
