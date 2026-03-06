import pytest

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery._base import _ConstraintMixin
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator


@pytest.fixture
def temporal_graph():
    graph = UndirectedGraph(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]
    )
    temporal_ordering = {"A": 3, "B": 1, "C": 2, "D": 0}
    return graph, temporal_ordering


@pytest.mark.parametrize("estimator_class", [BaseConstraintEstimator, _ConstraintMixin])
class TestGetPotentialSepsets:
    def test_temporal_ordering_filters_correctly(self, estimator_class, temporal_graph):
        graph, temporal_ordering = temporal_graph
        result = list(
            estimator_class._get_potential_sepsets(
                u="A",
                v="B",
                temporal_ordering=temporal_ordering,
                graph=graph,
                lim_neighbors=1,
            )
        )

        assert sorted(result) == [("D",), ("D",)]

    def test_temporal_ordering_symmetric(self, estimator_class, temporal_graph):
        graph, temporal_ordering = temporal_graph
        result_ab = list(
            estimator_class._get_potential_sepsets(
                u="A",
                v="B",
                temporal_ordering=temporal_ordering,
                graph=graph,
                lim_neighbors=1,
            )
        )
        result_ba = list(
            estimator_class._get_potential_sepsets(
                u="B",
                v="A",
                temporal_ordering=temporal_ordering,
                graph=graph,
                lim_neighbors=1,
            )
        )

        assert set(result_ab) == set(result_ba)

    def test_no_temporal_ordering(self, estimator_class, temporal_graph):
        graph, _ = temporal_graph
        result = list(
            estimator_class._get_potential_sepsets(
                u="A", v="B", temporal_ordering={}, graph=graph, lim_neighbors=1
            )
        )

        assert sorted(result) == [("C",), ("C",), ("D",), ("D",)]
