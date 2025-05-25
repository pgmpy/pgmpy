import unittest

import pandas as pd
import networkx as nx

from pgmpy.estimators import ExpertInLoop, ExpertKnowledge
from pgmpy.utils import get_example_model


class TestExpertInLoopWithExpertKnowledge(unittest.TestCase):
    def setUp(self):
        # Get example data
        self.model = get_example_model("cancer")
        self.data = self.model.simulate(int(1e3))

        # Define expert knowledge
        self.expert = ExpertKnowledge(
            forbidden_edges=[("Xray", "Smoker")],
            required_edges=[("Smoker", "Cancer")],
            temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Dyspnoea", "Xray"]],
        )

        # Create estimator with expert knowledge
        self.estimator = ExpertInLoop(data=self.data, expert_knowledge=self.expert)

    def test_forbidden_edges(self):
        """Test that forbidden edges are never added to the graph."""
        dag = self.estimator.estimate(effect_size_threshold=0.01, show_progress=False)

        # Check that forbidden edge is not present in either direction
        self.assertFalse(dag.has_edge("Xray", "Smoker"))
        self.assertFalse(dag.has_edge("Smoker", "Xray"))

    def test_required_edges(self):
        """Test that required edges are initialized and protected."""
        dag = self.estimator.estimate(effect_size_threshold=0.01, show_progress=False)

        # Check that required edge is present
        self.assertTrue(dag.has_edge("Smoker", "Cancer"))

        # Check that required edge is not reversed
        self.assertFalse(dag.has_edge("Cancer", "Smoker"))

    def test_temporal_order(self):
        """Test that temporal order influences edge directions."""
        dag = self.estimator.estimate(effect_size_threshold=0.01, show_progress=False)

        # Check that edges respect temporal order
        for u, v in dag.edges():
            if (
                u in self.expert.temporal_ordering
                and v in self.expert.temporal_ordering
            ):
                self.assertLess(
                    self.expert.temporal_ordering[u],
                    self.expert.temporal_ordering[v],
                    f"Edge {u}->{v} violates temporal order",
                )

    def test_no_expert_knowledge(self):
        """Test that behavior is unchanged when no expert knowledge is provided."""
        # Create estimator without expert knowledge
        estimator_no_expert = ExpertInLoop(data=self.data)

        # Run estimation
        dag_no_expert = estimator_no_expert.estimate(
            effect_size_threshold=0.01, show_progress=False
        )

        # Run estimation with expert knowledge
        dag_with_expert = self.estimator.estimate(
            effect_size_threshold=0.01, show_progress=False
        )

        # The graphs should be different since expert knowledge constrains the structure
        self.assertNotEqual(set(dag_no_expert.edges()), set(dag_with_expert.edges()))

    def test_required_edges_protection(self):
        """Test that required edges are protected from removal unless strongly contradicted."""
        # Create a dataset that strongly suggests Cancer->Smoker
        data_contradict = self.data.copy()
        data_contradict["Smoker"] = data_contradict[
            "Cancer"
        ].copy()  # Make them perfectly correlated

        # Create estimator with required edge Smoker->Cancer
        estimator_contradict = ExpertInLoop(
            data=data_contradict,
            expert_knowledge=ExpertKnowledge(required_edges=[("Smoker", "Cancer")]),
        )

        # Run estimation with very strict thresholds
        dag = estimator_contradict.estimate(
            effect_size_threshold=0.9,  # Very high threshold
            pval_threshold=0.01,  # Very strict p-value
            show_progress=False,
        )

        # Required edge should still be present despite data suggesting otherwise
        self.assertTrue(dag.has_edge("Smoker", "Cancer"))
        self.assertFalse(dag.has_edge("Cancer", "Smoker"))

    def test_temporal_order_resolution(self):
        """Test that temporal order is used to resolve edge directions before consulting orientation_fn."""

        # Create a custom orientation function that would reverse temporal order
        def reverse_orientation(var1, var2, **kwargs):
            return (var2, var1)  # Always reverse the order

        # Run estimation with the reversing orientation function
        dag = self.estimator.estimate(
            effect_size_threshold=0.01,
            orientation_fn=reverse_orientation,
            show_progress=False,
        )

        # Check that temporal order still takes precedence
        for u, v in dag.edges():
            if (
                u in self.expert.temporal_ordering
                and v in self.expert.temporal_ordering
            ):
                self.assertLess(
                    self.expert.temporal_ordering[u],
                    self.expert.temporal_ordering[v],
                    f"Edge {u}->{v} violates temporal order despite orientation function",
                )
