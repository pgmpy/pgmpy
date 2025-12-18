#!/usr/bin/env python

from unittest.mock import Mock

import numpy as np
import pytest

from pgmpy.bandits.base import (
    CausalBanditLearner,
    CausalBanditModel,
    CausalBanditPolicy,
)
from pgmpy.base import DAG


class TestCausalBanditModel:
    """Test CausalBanditModel class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test graph: X -> Y -> Z
        self.graph = DAG([("X", "Y"), ("Y", "Z")])
        self.action_variables = ["X"]
        self.outcome_variable = "Z"

    def test_init_valid_graph(self):
        """Test initialization with valid graph."""
        model = CausalBanditModel(
            causal_graph=self.graph,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            seed=42,
        )

        assert model.graph == self.graph
        assert model.action_variables == self.action_variables
        assert model.outcome_variable == self.outcome_variable

    def test_init_invalid_outcome_variable(self):
        """Test initialization with invalid outcome variable."""
        with pytest.raises(ValueError, match="Outcome variable"):
            CausalBanditModel(
                causal_graph=self.graph,
                action_variables=self.action_variables,
                outcome_variable="Invalid",
            )

    def test_init_invalid_action_variable(self):
        """Test initialization with invalid action variable."""
        with pytest.raises(ValueError, match="Action variable"):
            CausalBanditModel(
                causal_graph=self.graph,
                action_variables=["Invalid"],
                outcome_variable=self.outcome_variable,
            )

    def test_get_action_space(self):
        """Test getting action space."""
        model = CausalBanditModel(
            causal_graph=self.graph,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
        )

        action_space = model.get_action_space()
        assert action_space == self.action_variables
        assert id(action_space) != id(self.action_variables)  # Should be a copy

    def test_get_valid_interventions(self):
        """Test getting valid interventions."""
        model = CausalBanditModel(
            causal_graph=self.graph,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
        )

        interventions = model.get_valid_interventions()

        # Should include: {}, {X: 0}, {X: 1}
        assert len(interventions) == 3
        assert {} in interventions
        assert {"X": 0} in interventions
        assert {"X": 1} in interventions

    def test_simulate_intervention(self):
        """Test intervention simulation."""
        model = CausalBanditModel(
            causal_graph=self.graph,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
            seed=42,
        )

        # Test with intervention
        reward1 = model.simulate_intervention({"X": 1})
        assert isinstance(reward1, float)

        # Test with no intervention
        reward2 = model.simulate_intervention({})
        assert isinstance(reward2, float)

        # Test with context
        reward3 = model.simulate_intervention({"X": 1}, context={"Y": 0.5})
        assert isinstance(reward3, float)

    def test_get_causal_effect(self):
        """Test causal effect estimation."""
        model = CausalBanditModel(
            causal_graph=self.graph,
            action_variables=self.action_variables,
            outcome_variable=self.outcome_variable,
        )

        # Test with intervention
        effect1 = model.get_causal_effect({"X": 1})
        assert isinstance(effect1, float)

        # Test with no intervention
        effect2 = model.get_causal_effect({})
        assert effect2 == 0.0


class MockPolicy(CausalBanditPolicy):
    """Mock policy for testing."""

    def __init__(self, action_space, **kwargs):
        super().__init__(action_space, **kwargs)
        self.selected_actions = []

    def select_action(self, context=None):
        self.t += 1
        action = np.random.randint(0, self.n_actions)
        self.selected_actions.append(action)
        return action

    def update(self, action_idx, reward, context=None):
        pass


class TestCausalBanditLearner:
    """Test CausalBanditLearner class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple environment
        graph = DAG([("X", "Y")])
        self.environment = CausalBanditModel(
            causal_graph=graph, action_variables=["X"], outcome_variable="Y", seed=42
        )

        # Create mock policy
        action_space = self.environment.get_valid_interventions()
        self.policy = MockPolicy(action_space)

    def test_init(self):
        """Test learner initialization."""
        learner = CausalBanditLearner(
            environment=self.environment, policy=self.policy, horizon=100
        )

        assert learner.environment == self.environment
        assert learner.policy == self.policy
        assert learner.horizon == 100
        assert len(learner.history["actions"]) == 0

    def test_run_basic(self):
        """Test basic learning run."""
        learner = CausalBanditLearner(
            environment=self.environment, policy=self.policy, horizon=10
        )

        history = learner.run()

        assert len(history["actions"]) == 10
        assert len(history["rewards"]) == 10
        assert len(history["contexts"]) == 10
        assert len(history["regrets"]) == 10
        assert len(history["cumulative_regret"]) == 10

    def test_run_with_context_generator(self):
        """Test learning run with context generator."""

        def context_gen(t):
            return {"context_var": t * 0.1}

        learner = CausalBanditLearner(
            environment=self.environment, policy=self.policy, horizon=5
        )

        history = learner.run(context_generator=context_gen)

        assert len(history["actions"]) == 5
        for i, context in enumerate(history["contexts"]):
            assert context["context_var"] == i * 0.1

    def test_run_with_oracle_policy(self):
        """Test learning run with oracle policy for regret calculation."""

        def oracle_policy(context):
            return 1  # Always select action 1

        learner = CausalBanditLearner(
            environment=self.environment, policy=self.policy, horizon=5
        )

        history = learner.run(oracle_policy=oracle_policy)

        assert len(history["regrets"]) == 5
        for regret in history["regrets"]:
            assert isinstance(regret, float)

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        learner = CausalBanditLearner(
            environment=self.environment, policy=self.policy, horizon=10
        )

        # Run to generate history
        learner.run()

        metrics = learner.get_performance_metrics()

        assert "cumulative_regret" in metrics
        assert "average_reward" in metrics
        assert "total_reward" in metrics
        assert "regret_per_round" in metrics

        assert isinstance(metrics["average_reward"], float)
        assert isinstance(metrics["total_reward"], (int, float))

    def test_get_performance_metrics_empty_history(self):
        """Test performance metrics with empty history."""
        learner = CausalBanditLearner(
            environment=self.environment, policy=self.policy, horizon=10
        )

        metrics = learner.get_performance_metrics()
        assert metrics == {}


class TestCausalBanditPolicy:
    """Test abstract CausalBanditPolicy class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract policy cannot be instantiated."""
        with pytest.raises(TypeError):
            CausalBanditPolicy([{}])

    def test_mock_policy_basic_functionality(self):
        """Test that mock policy works correctly."""
        action_space = [{}, {"X": 0}, {"X": 1}]
        policy = MockPolicy(action_space)

        assert policy.n_actions == 3
        assert policy.t == 0

        # Test action selection
        action = policy.select_action()
        assert 0 <= action < 3
        assert policy.t == 1

        # Test update
        policy.update(action, 1.0)

        # Test reset
        policy.reset()
        assert policy.t == 0
