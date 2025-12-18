#!/usr/bin/env python

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .base import CausalBanditPolicy


class EpsilonGreedyCausalBandit(CausalBanditPolicy):
    """
    Epsilon-greedy policy for causal bandits.

    Explores with probability epsilon, otherwise selects
    the action with highest estimated causal effect.
    """

    def __init__(
        self,
        action_space: List[Dict[str, Any]],
        epsilon: float = 0.1,
        decay_rate: float = 0.0,
        **kwargs,
    ):
        """
        Initialize epsilon-greedy policy.

        Parameters
        ----------
        action_space : list of dict
            Available actions/interventions
        epsilon : float
            Exploration probability
        decay_rate : float
            Rate at which epsilon decays over time
        """
        super().__init__(action_space, **kwargs)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate

        # Track estimates and counts
        self.action_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

    def select_action(self, context: Optional[Dict[str, Any]] = None) -> int:
        """Select action using epsilon-greedy strategy."""
        self.t += 1

        # Decay epsilon
        if self.decay_rate > 0:
            self.epsilon = self.initial_epsilon * np.exp(-self.decay_rate * self.t)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action so far
            return np.argmax(self.action_values)

    def update(
        self, action_idx: int, reward: float, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update action value estimates."""
        self.action_counts[action_idx] += 1

        # Incremental mean update
        n = self.action_counts[action_idx]
        self.action_values[action_idx] += (reward - self.action_values[action_idx]) / n

    def reset(self) -> None:
        """Reset policy state."""
        super().reset()
        self.epsilon = self.initial_epsilon
        self.action_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)


class UCBCausalBandit(CausalBanditPolicy):
    """
    Upper Confidence Bound (UCB) policy for causal bandits.

    Balances exploitation and exploration by selecting actions
    with highest upper confidence bound.
    """

    def __init__(self, action_space: List[Dict[str, Any]], c: float = 1.0, **kwargs):
        """
        Initialize UCB policy.

        Parameters
        ----------
        action_space : list of dict
            Available actions/interventions
        c : float
            Exploration parameter
        """
        super().__init__(action_space, **kwargs)
        self.c = c

        # Track estimates and counts
        self.action_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

    def select_action(self, context: Optional[Dict[str, Any]] = None) -> int:
        """Select action using UCB strategy."""
        self.t += 1

        # If any action hasn't been tried, select it
        if np.any(self.action_counts == 0):
            unexplored = np.where(self.action_counts == 0)[0]
            return unexplored[0]

        # Calculate UCB values
        ucb_values = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            confidence_width = self.c * math.sqrt(
                2 * math.log(self.t) / self.action_counts[i]
            )
            ucb_values[i] = self.action_values[i] + confidence_width

        return np.argmax(ucb_values)

    def update(
        self, action_idx: int, reward: float, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update action value estimates."""
        self.action_counts[action_idx] += 1

        # Incremental mean update
        n = self.action_counts[action_idx]
        self.action_values[action_idx] += (reward - self.action_values[action_idx]) / n

    def reset(self) -> None:
        """Reset policy state."""
        super().reset()
        self.action_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)


class ThompsonSamplingCausalBandit(CausalBanditPolicy):
    """
    Thompson Sampling policy for causal bandits.

    Uses Bayesian approach with Beta-Bernoulli or Normal-Normal
    conjugate priors for reward estimation.
    """

    def __init__(
        self,
        action_space: List[Dict[str, Any]],
        prior_type: str = "normal",
        prior_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize Thompson Sampling policy.

        Parameters
        ----------
        action_space : list of dict
            Available actions/interventions
        prior_type : str
            Type of prior distribution ("normal", "beta")
        prior_params : dict, optional
            Prior distribution parameters
        """
        super().__init__(action_space, **kwargs)
        self.prior_type = prior_type

        if prior_params is None:
            if prior_type == "beta":
                prior_params = {"alpha": 1.0, "beta": 1.0}
            elif prior_type == "normal":
                prior_params = {"mu": 0.0, "sigma": 1.0, "nu": 1.0}
            else:
                raise ValueError(f"Unsupported prior type: {prior_type}")

        self.prior_params = prior_params

        # Initialize posterior parameters
        if prior_type == "beta":
            # Beta-Bernoulli conjugate
            self.alpha = np.full(self.n_actions, prior_params["alpha"])
            self.beta = np.full(self.n_actions, prior_params["beta"])
        elif prior_type == "normal":
            # Normal-Normal conjugate
            self.mu = np.full(self.n_actions, prior_params["mu"])
            self.sigma_squared = np.full(self.n_actions, prior_params["sigma"] ** 2)
            self.nu = np.full(self.n_actions, prior_params["nu"])
        else:
            raise ValueError(f"Unsupported prior type: {prior_type}")

    def select_action(self, context: Optional[Dict[str, Any]] = None) -> int:
        """Select action using Thompson Sampling."""
        self.t += 1

        # Sample from posterior distributions
        if self.prior_type == "beta":
            samples = np.array(
                [
                    np.random.beta(self.alpha[i], self.beta[i])
                    for i in range(self.n_actions)
                ]
            )
        elif self.prior_type == "normal":
            samples = np.array(
                [
                    np.random.normal(
                        self.mu[i], math.sqrt(self.sigma_squared[i] / self.nu[i])
                    )
                    for i in range(self.n_actions)
                ]
            )

        return np.argmax(samples)

    def update(
        self, action_idx: int, reward: float, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update posterior distributions."""
        if self.prior_type == "beta":
            # Assume reward is binary (0 or 1)
            if reward > 0:
                self.alpha[action_idx] += 1
            else:
                self.beta[action_idx] += 1

        elif self.prior_type == "normal":
            # Normal-Normal conjugate update
            old_nu = self.nu[action_idx]
            old_mu = self.mu[action_idx]

            self.nu[action_idx] += 1
            self.mu[action_idx] = (old_nu * old_mu + reward) / self.nu[action_idx]

    def reset(self) -> None:
        """Reset policy state."""
        super().reset()

        if self.prior_type == "beta":
            self.alpha = np.full(self.n_actions, self.prior_params["alpha"])
            self.beta = np.full(self.n_actions, self.prior_params["beta"])
        elif self.prior_type == "normal":
            self.mu = np.full(self.n_actions, self.prior_params["mu"])
            self.sigma_squared = np.full(
                self.n_actions, self.prior_params["sigma"] ** 2
            )
            self.nu = np.full(self.n_actions, self.prior_params["nu"])


class ContextualCausalBandit(CausalBanditPolicy):
    """
    Contextual causal bandit using linear regression for reward modeling.

    Models the causal effect as a linear function of context and actions.
    """

    def __init__(
        self,
        action_space: List[Dict[str, Any]],
        context_dim: int,
        alpha: float = 1.0,
        **kwargs,
    ):
        """
        Initialize contextual policy.

        Parameters
        ----------
        action_space : list of dict
            Available actions/interventions
        context_dim : int
            Dimensionality of context vectors
        alpha : float
            Regularization parameter
        """
        super().__init__(action_space, **kwargs)
        self.context_dim = context_dim
        self.alpha = alpha

        # Linear model parameters (one per action)
        self.A = [self.alpha * np.eye(context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(self.n_actions)]

    def _context_to_vector(self, context: Optional[Dict[str, Any]]) -> np.ndarray:
        """Convert context dict to feature vector."""
        if context is None:
            return np.ones(self.context_dim)

        # Simple conversion - in practice, this would be more sophisticated
        features = []
        for key in sorted(context.keys()):
            if isinstance(context[key], (int, float)):
                features.append(context[key])
            else:
                features.append(1.0 if context[key] else 0.0)

        # Pad or truncate to context_dim
        features = features[: self.context_dim]
        while len(features) < self.context_dim:
            features.append(0.0)

        return np.array(features)

    def select_action(self, context: Optional[Dict[str, Any]] = None) -> int:
        """Select action using linear UCB."""
        self.t += 1

        x = self._context_to_vector(context)

        ucb_values = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]

            # Upper confidence bound
            confidence_width = self.alpha * math.sqrt(x.T @ A_inv @ x)
            ucb = x.T @ theta_hat + confidence_width
            ucb_values.append(ucb)

        return np.argmax(ucb_values)

    def update(
        self, action_idx: int, reward: float, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update linear model parameters."""
        x = self._context_to_vector(context)

        self.A[action_idx] += np.outer(x, x)
        self.b[action_idx] += reward * x

    def reset(self) -> None:
        """Reset policy state."""
        super().reset()
        self.A = [self.alpha * np.eye(self.context_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_actions)]
