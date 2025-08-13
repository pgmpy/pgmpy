import unittest
from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy.base import UndirectedGraph
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator
from pgmpy.independencies import Independencies


class TestBaseConstraintEstimator(unittest.TestCase):
    @staticmethod
    def fake_ci_t(X, Y, Z=None, **kwargs):
        """
        Simple fake CI test for deterministic behavior.
        Independence rules:
          1. A ⫫ B  (unconditional)
          2. A ⫫ C | B
        Everything else: dependent.
        """
        if Z is None:
            Z = []
        Z = list(Z)
        if (X, Y) == ("A", "B") or (Y, X) == ("A", "B"):
            return True
        if (X, Y) == ("A", "C") and Z == ["B"]:
            return True
        if (Y, X) == ("A", "C") and Z == ["B"]:
            return True
        return False

    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "A": [0, 1, 0, 1, 0, 1, 0, 1],
                "B": [0, 0, 1, 1, 0, 0, 1, 1],
                "C": [0, 1, 1, 0, 1, 0, 0, 1],
                "D": [1, 1, 0, 0, 1, 1, 0, 0],
            }
        )
        self.sample = Independencies(["A", "B"], ["C", "D"])
        self.estimator = BaseConstraintEstimator(data=self.sample_data)

    def test_init_with_data(self):
        estimator = BaseConstraintEstimator(data=self.sample_data)
        self.assertIsNotNone(estimator.data)
        self.assertEqual(list(estimator.data.columns), ["A", "B", "C", "D"])

    def test_init_with_independencies(self):
        estimator = BaseConstraintEstimator(independencies=self.sample)
        self.assertIsNone(estimator.data)
        self.assertIsNotNone(estimator.independencies)

    def test_get_potential_sepsets_basic(self):
        graph = UndirectedGraph()
        graph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
            ]
        )
        sepsets = list(
            self.estimator._get_potential_sepsets(
                "A",
                "B",
                {},
                graph,
                1,
            )
        )
        self.assertEqual(sepsets, [("C",)])

    def test_get_potential_sepsets_complete_graph(self):
        graph = nx.complete_graph(
            ["A", "B", "C", "D"],
            create_using=nx.Graph,
        )
        sepsets = list(
            self.estimator._get_potential_sepsets(
                "A",
                "B",
                {},
                graph,
                1,
            )
        )
        self.assertEqual(len(sepsets), 4)

    def test_get_potential_sepsets_with_temporal_ordering(self):
        graph = nx.complete_graph(
            ["A", "B", "C", "D"],
            create_using=nx.Graph,
        )
        temporal_ordering = {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
        }
        sepsets = list(
            self.estimator._get_potential_sepsets(
                "A",
                "B",
                temporal_ordering,
                graph,
                1,
            )
        )
        self.assertEqual(sepsets, [])

    def test_get_potential_sepsets_no_neighbors(self):
        graph = UndirectedGraph()
        graph.add_edge("A", "B")
        sepsets = list(
            self.estimator._get_potential_sepsets(
                "A",
                "B",
                {},
                graph,
                1,
            )
        )
        self.assertEqual(sepsets, [])

    def test_build_skeleton_no_independence(self):
        graph, separating_sets = self.estimator.build_skeleton(
            ci_test=lambda *a, **k: False,
            max_cond_vars=1,
            variant="orig",
            show_progress=False,
        )
        expected_edges = len(list(combinations(self.sample_data.columns, 2)))
        self.assertEqual(len(graph.edges()), expected_edges)
        self.assertEqual(len(separating_sets), 0)

    def test_build_skeleton_with_independence(self):
        graph, separating_sets = self.estimator.build_skeleton(
            ci_test=self.fake_ci_t,
            max_cond_vars=1,
            variant="stable",
            show_progress=False,
        )
        self.assertFalse(graph.has_edge("A", "B"))
        self.assertIn(frozenset(["A", "B"]), separating_sets)
        self.assertEqual(
            separating_sets[frozenset(["A", "B"])],
            (),
        )

    def test_build_skeleton_max_cond_vars(self):
        graph, _ = self.estimator.build_skeleton(
            ci_test=lambda *a, **k: False,
            max_cond_vars=0,
            variant="orig",
            show_progress=False,
        )
        # Just check it's a valid nx.Graph
        self.assertIsInstance(graph, nx.Graph)

    def test_build_skeleton_return_types(self):
        result = self.estimator.build_skeleton(
            ci_test=lambda *a, **k: False,
            max_cond_vars=1,
            variant="parallel",
            show_progress=False,
        )
        graph, separating_sets = result
        self.assertIsInstance(graph, nx.Graph)
        self.assertIsInstance(separating_sets, dict)

    def test_build_skeleton_separating_sets_format(self):
        def fake_with_sep(X, Y, Z=None, **kwargs):
            if Z is None:
                Z = []
            return (X, Y) == ("A", "B") and tuple(Z) == ("C",)

        graph, separating_sets = self.estimator.build_skeleton(
            ci_test=fake_with_sep,
            max_cond_vars=1,
            variant="orig",
            show_progress=False,
        )
        for edge_set, sep_set in separating_sets.items():
            self.assertIsInstance(edge_set, frozenset)
            self.assertEqual(len(edge_set), 2)
            self.assertIsInstance(sep_set, tuple)

    def tearDown(self):
        del self.sample_data
        del self.sample_independencies
        del self.estimator
