#!/usr/bin/env python

import pytest
import numpy as np

from pgmpy.bandits.causal_bandits import (
    EpsilonGreedyCausalBandit,
    UCBCausalBandit,
    ThompsonSamplingCausalBandit,
    ContextualCausalBandit
)


class TestEpsilonGreedyCausalBandit:
    """Test EpsilonGreedyCausalBandit class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = [{}, {"X": 0}, {"X": 1}]

    def test_init(self):
        """Test initialization."""
        policy = EpsilonGreedyCausalBandit(
            action_space=self.action_space,
            epsilon=0.2,
            decay_rate=0.01
        )

        assert policy.epsilon == 0.2
        assert policy.initial_epsilon == 0.2
        assert policy.decay_rate == 0.01
        assert policy.n_actions == 3
        assert len(policy.action_values) == 3
        assert len(policy.action_counts) == 3

    def test_select_action_pure_exploration(self):
        """Test action selection with epsilon=1 (pure exploration)."""
        policy = EpsilonGreedyCausalBandit(
            action_space=self.action_space,
            epsilon=1.0  # Always explore
        )

        # Should select random actions
        actions = [policy.select_action() for _ in range(100)]
        assert all(0 <= action < 3 for action in actions)
        assert policy.t == 100

    def test_select_action_pure_exploitation(self):
        """Test action selection with epsilon=0 (pure exploitation)."""
        policy = EpsilonGreedyCausalBandit(
            action_space=self.action_space,
            epsilon=0.0  # Never explore
        )

        # Update to create a clear best action
        policy.update(1, 10.0)  # Make action 1 clearly best

        actions = [policy.select_action() for _ in range(10)]
        assert all(action == 1 for action in actions)

    def test_update(self):
        """Test action value updates."""
        policy = EpsilonGreedyCausalBandit(action_space=self.action_space)

        # Update action 0 multiple times
        rewards = [1.0, 2.0, 3.0]
        for reward in rewards:
            policy.update(0, reward)

        expected_avg = sum(rewards) / len(rewards)
        assert abs(policy.action_values[0] - expected_avg) < 1e-10
        assert policy.action_counts[0] == 3

    def test_epsilon_decay(self):
        """Test epsilon decay over time."""
        policy = EpsilonGreedyCausalBandit(
            action_space=self.action_space,
            epsilon=1.0,
            decay_rate=0.1
        )

        initial_epsilon = policy.epsilon
        for _ in range(10):
            policy.select_action()

        assert policy.epsilon < initial_epsilon

    def test_reset(self):
        """Test policy reset."""
        policy = EpsilonGreedyCausalBandit(
            action_space=self.action_space,
            epsilon=0.1,
            decay_rate=0.01
        )

        # Make some updates
        policy.select_action()
        policy.update(0, 5.0)

        # Reset
        policy.reset()

        assert policy.t == 0
        assert policy.epsilon == 0.1  # Back to initial
        assert np.allclose(policy.action_values, 0.0)
        assert np.allclose(policy.action_counts, 0.0)


class TestUCBCausalBandit:
    """Test UCBCausalBandit class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = [{}, {"X": 0}, {"X": 1}]

    def test_init(self):
        """Test initialization."""
        policy = UCBCausalBandit(
            action_space=self.action_space,
            c=2.0
        )

        assert policy.c == 2.0
        assert policy.n_actions == 3
        assert len(policy.action_values) == 3
        assert len(policy.action_counts) == 3

    def test_select_action_unexplored_actions(self):
        """Test that unexplored actions are selected first."""
        policy = UCBCausalBandit(action_space=self.action_space)

        # All actions should be selected before any is repeated
        selected = set()
        for _ in range(3):
            action = policy.select_action()
            selected.add(action)
            # Simulate update to mark action as explored
            policy.update(action, 1.0)

        assert len(selected) == 3  # All actions selected

    def test_select_action_ucb_calculation(self):
        """Test UCB calculation for action selection."""
        policy = UCBCausalBandit(action_space=self.action_space, c=1.0)

        # Initialize all actions
        for i in range(3):
            policy.select_action()
            policy.update(i, i)  # Give different rewards

        # Now UCB should guide selection
        action = policy.select_action()
        assert 0 <= action < 3

    def test_update(self):
        """Test action value updates."""
        policy = UCBCausalBandit(action_space=self.action_space)

        rewards = [1.0, 3.0, 2.0]
        for reward in rewards:
            policy.update(1, reward)

        expected_avg = sum(rewards) / len(rewards)
        assert abs(policy.action_values[1] - expected_avg) < 1e-10
        assert policy.action_counts[1] == 3

    def test_reset(self):
        """Test policy reset."""
        policy = UCBCausalBandit(action_space=self.action_space)

        # Make some updates
        policy.select_action()
        policy.update(0, 5.0)

        # Reset
        policy.reset()

        assert policy.t == 0
        assert np.allclose(policy.action_values, 0.0)
        assert np.allclose(policy.action_counts, 0.0)


class TestThompsonSamplingCausalBandit:
    """Test ThompsonSamplingCausalBandit class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = [{}, {"X": 0}, {"X": 1}]

    def test_init_beta_prior(self):
        """Test initialization with Beta prior."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="beta",
            prior_params={"alpha": 2.0, "beta": 3.0}
        )

        assert policy.prior_type == "beta"
        assert np.allclose(policy.alpha, 2.0)
        assert np.allclose(policy.beta, 3.0)

    def test_init_normal_prior(self):
        """Test initialization with Normal prior."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="normal",
            prior_params={"mu": 1.0, "sigma": 0.5, "nu": 2.0}
        )

        assert policy.prior_type == "normal"
        assert np.allclose(policy.mu, 1.0)
        assert policy.prior_params["sigma"] == 0.5
        assert np.allclose(policy.nu, 2.0)

    def test_init_invalid_prior(self):
        """Test initialization with invalid prior type."""
        with pytest.raises(ValueError, match="Unsupported prior type"):
            ThompsonSamplingCausalBandit(
                action_space=self.action_space,
                prior_type="invalid"
            )

    def test_select_action_beta(self):
        """Test action selection with Beta prior."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="beta"
        )

        actions = [policy.select_action() for _ in range(10)]
        assert all(0 <= action < 3 for action in actions)

    def test_select_action_normal(self):
        """Test action selection with Normal prior."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="normal"
        )

        actions = [policy.select_action() for _ in range(10)]
        assert all(0 <= action < 3 for action in actions)

    def test_update_beta_prior(self):
        """Test updates with Beta prior."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="beta",
            prior_params={"alpha": 1.0, "beta": 1.0}
        )

        # Simulate positive reward (should increase alpha)
        initial_alpha = policy.alpha[0]
        policy.update(0, 1.0)
        assert policy.alpha[0] == initial_alpha + 1

        # Simulate negative reward (should increase beta)
        initial_beta = policy.beta[1]
        policy.update(1, 0.0)
        assert policy.beta[1] == initial_beta + 1

    def test_update_normal_prior(self):
        """Test updates with Normal prior."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="normal",
            prior_params={"mu": 0.0, "sigma": 1.0, "nu": 1.0}
        )

        # Update with reward
        initial_nu = policy.nu[0]
        policy.update(0, 2.0)

        assert policy.nu[0] == initial_nu + 1
        # Posterior mean should move towards observed reward

    def test_reset(self):
        """Test policy reset."""
        policy = ThompsonSamplingCausalBandit(
            action_space=self.action_space,
            prior_type="beta",
            prior_params={"alpha": 2.0, "beta": 3.0}
        )

        # Make updates
        policy.select_action()
        policy.update(0, 1.0)

        # Reset
        policy.reset()

        assert policy.t == 0
        assert np.allclose(policy.alpha, 2.0)
        assert np.allclose(policy.beta, 3.0)


class TestContextualCausalBandit:
    """Test ContextualCausalBandit class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = [{}, {"X": 0}, {"X": 1}]
        self.context_dim = 3

    def test_init(self):
        """Test initialization."""
        policy = ContextualCausalBandit(
            action_space=self.action_space,
            context_dim=self.context_dim,
            alpha=2.0
        )

        assert policy.context_dim == self.context_dim
        assert policy.alpha == 2.0
        assert len(policy.A) == 3  # One per action
        assert len(policy.b) == 3

    def test_context_to_vector(self):
        """Test context conversion to vector."""
        policy = ContextualCausalBandit(
            action_space=self.action_space,
            context_dim=self.context_dim
        )

        # Test with None context
        vec1 = policy._context_to_vector(None)
        assert len(vec1) == self.context_dim
        assert np.allclose(vec1, 1.0)

        # Test with dict context
        context = {"feature1": 1.5, "feature2": -0.5}
        vec2 = policy._context_to_vector(context)
        assert len(vec2) == self.context_dim

    def test_select_action(self):
        """Test contextual action selection."""
        policy = ContextualCausalBandit(
            action_space=self.action_space,
            context_dim=self.context_dim
        )

        context = {"feature1": 1.0, "feature2": 0.5}
        action = policy.select_action(context)
        assert 0 <= action < 3

    def test_update(self):
        """Test parameter updates."""
        policy = ContextualCausalBandit(
            action_space=self.action_space,
            context_dim=self.context_dim
        )

        context = {"feature1": 1.0, "feature2": -0.5}

        # Store initial parameters
        initial_A = policy.A[0].copy()
        initial_b = policy.b[0].copy()

        # Update
        policy.update(0, 2.0, context)

        # Parameters should have changed
        assert not np.allclose(policy.A[0], initial_A)
        assert not np.allclose(policy.b[0], initial_b)

    def test_reset(self):
        """Test policy reset."""
        policy = ContextualCausalBandit(
            action_space=self.action_space,
            context_dim=self.context_dim,
            alpha=1.5
        )

        # Make updates
        context = {"feature1": 1.0}
        policy.select_action(context)
        policy.update(0, 1.0, context)

        # Reset
        policy.reset()

        assert policy.t == 0
        # Check that A matrices are back to alpha * I
        for A in policy.A:
            expected = 1.5 * np.eye(self.context_dim)
            assert np.allclose(A, expected)
        # Check that b vectors are back to zeros
        for b in policy.b:
            assert np.allclose(b, 0.0)