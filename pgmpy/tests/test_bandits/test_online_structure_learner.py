#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.bandits.online_structure_learner import OnlineCausalStructureLearner


class TestOnlineCausalStructureLearner:
    """Test OnlineCausalStructureLearner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.variables = ["X", "Y", "Z"]
        self.action_variables = ["X"]
        self.outcome_variable = "Z"

    def test_init(self):
        """Test initialization."""
        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable
        )

        assert learner.variables == self.variables
        assert learner.action_variables == self.action_variables
        assert learner.outcome_variable == self.outcome_variable
        assert len(learner.current_graph.nodes()) == 3
        assert learner.total_observations == 0

    def test_init_with_initial_graph(self):
        """Test initialization with initial graph."""
        initial_graph = DAG([("X", "Y"), ("Y", "Z")])

        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            initial_graph=initial_graph
        )

        assert learner.current_graph.number_of_edges() == 2
        assert learner.current_graph.has_edge("X", "Y")
        assert learner.current_graph.has_edge("Y", "Z")

    def test_add_observation(self):
        """Test adding observations."""
        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            update_frequency=10  # Don't update during test
        )

        # Add observation
        intervention = {"X": 1}
        outcome = 0.5
        context = {"Y": 0.3}

        learner.add_observation(intervention, outcome, context)

        assert learner.total_observations == 1
        assert len(learner.data_buffer) == 1

        observation = learner.data_buffer[0]
        assert observation["X"] == 1
        assert observation["Z"] == 0.5
        assert observation["Y"] == 0.3

    def test_add_observation_missing_variables(self):
        """Test adding observation with missing variables filled as NaN."""
        learner = OnlineCausalStructureLearner(
            variables=self.variables + ["W"],  # Extra variable
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            update_frequency=10
        )

        intervention = {"X": 1}
        outcome = 0.5

        learner.add_observation(intervention, outcome)

        observation = learner.data_buffer[0]
        assert observation["X"] == 1
        assert observation["Z"] == 0.5
        assert pd.isna(observation["Y"])  # Missing variable should be NaN
        assert pd.isna(observation["W"])  # Missing variable

    def test_get_current_graph(self):
        """Test getting current graph."""
        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable
        )

        graph = learner.get_current_graph()
        assert isinstance(graph, DAG)
        assert graph.number_of_nodes() == 3

    def test_get_causal_parents(self):
        """Test getting causal parents."""
        initial_graph = DAG([("X", "Y"), ("Y", "Z")])

        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            initial_graph=initial_graph
        )

        # Test parents of Z
        parents = learner.get_causal_parents("Z")
        assert parents == ["Y"]

        # Test parents of Y
        parents = learner.get_causal_parents("Y")
        assert parents == ["X"]

        # Test parents of X (should be empty)
        parents = learner.get_causal_parents("X")
        assert parents == []

    def test_get_causal_children(self):
        """Test getting causal children."""
        initial_graph = DAG([("X", "Y"), ("Y", "Z")])

        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            initial_graph=initial_graph
        )

        # Test children of X
        children = learner.get_causal_children("X")
        assert children == ["Y"]

        # Test children of Y
        children = learner.get_causal_children("Y")
        assert children == ["Z"]

        # Test children of Z (should be empty)
        children = learner.get_causal_children("Z")
        assert children == []

    def test_is_causal_path(self):
        """Test causal path detection."""
        initial_graph = DAG([("X", "Y"), ("Y", "Z")])

        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            initial_graph=initial_graph
        )

        # Test direct path
        assert learner.is_causal_path("X", "Y") == True
        assert learner.is_causal_path("Y", "Z") == True

        # Test indirect path
        assert learner.is_causal_path("X", "Z") == True

        # Test no path
        assert learner.is_causal_path("Z", "X") == False
        assert learner.is_causal_path("Z", "Y") == False

    def test_estimate_intervention_targets(self):
        """Test intervention target estimation."""
        initial_graph = DAG([("X", "Y"), ("Y", "Z")])

        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            initial_graph=initial_graph
        )

        # Test targets for Z (should include X since X->Y->Z)
        targets = learner.estimate_intervention_targets("Z")
        assert "X" in targets

        # Test targets for Y (should include X since X->Y)
        targets = learner.estimate_intervention_targets("Y")
        assert "X" in targets

    def test_get_learning_summary(self):
        """Test learning summary."""
        initial_graph = DAG([("X", "Y")])

        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            initial_graph=initial_graph
        )

        # Add some observations
        for i in range(5):
            learner.add_observation({"X": i % 2}, float(i))

        summary = learner.get_learning_summary()

        assert summary["total_observations"] == 5
        assert summary["n_nodes"] >= 2  # At least the initial graph nodes
        assert summary["n_edges"] == 1
        assert "current_edges" in summary
        assert "current_nodes" in summary

    def test_structure_update_with_sufficient_data(self):
        """Test structure update when enough data is available."""
        learner = OnlineCausalStructureLearner(
            variables=self.variables,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            update_frequency=5,  # Update every 5 observations
            score_type="bic"
        )

        # Generate synthetic data with clear structure
        np.random.seed(42)
        for i in range(20):  # More than update frequency
            x = np.random.randint(0, 2)
            y = x + np.random.normal(0, 0.1)  # Y depends on X
            z = y + np.random.normal(0, 0.1)  # Z depends on Y

            learner.add_observation({"X": x}, z, {"Y": y})

        # Should have triggered updates
        assert learner.update_count > 0
        assert learner.total_observations == 20