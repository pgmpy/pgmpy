from abc import ABC, abstractmethod

import numpy as np


class CausalBanditPolicy(ABC):
    """
    Abstract base class for causal bandit policies.

    A policy selects interventions (arms) on a ``CausalBanditModel`` and
    updates its internal state based on observed rewards.

    Parameters
    ----------
    model : pgmpy.bandits.CausalBanditModel
        The causal bandit environment.
    """

    def __init__(self, model):
        self.model = model
        self.actions = model.get_possible_interventions()
        self._key_to_action = {self._action_key(a): a for a in self.actions}

    @staticmethod
    def _action_key(action):
        """Return a hashable key for an action dict."""
        return tuple(sorted(action.items()))

    @abstractmethod
    def select_action(self):
        """Select an arm / intervention dict."""

    @abstractmethod
    def update(self, action, reward):
        """Update internal state after observing *reward* for *action*."""

    @abstractmethod
    def recommend(self):
        """Return the best arm found so far."""

    def fit(self, observational_data=None):
        """
        Optional warm-start from observational data.

        Parameters
        ----------
        observational_data : pd.DataFrame or None
            Observational samples from the model (no intervention).
        """


class CausalUCB(CausalBanditPolicy):
    """
    Upper Confidence Bound policy for causal bandits.

    Uses UCB1 exploration bonus on top of empirical mean rewards. Arms are
    initialised with zero counts and are tried round-robin before UCB scores
    are compared.

    Parameters
    ----------
    model : pgmpy.bandits.CausalBanditModel
    exploration_weight : float (default: ``1.0``)
        Multiplier on the UCB exploration term ``sqrt(ln(t) / n_i)``.
    """

    def __init__(self, model, exploration_weight=1.0):
        super().__init__(model)
        self.exploration_weight = exploration_weight
        self.counts = {self._action_key(a): 0 for a in self.actions}
        self.q_values = {self._action_key(a): 0.0 for a in self.actions}
        self._total_counts = 0

    def select_action(self):
        # Round-robin until every arm tried once
        for key, action in self._key_to_action.items():
            if self.counts[key] == 0:
                return action

        # UCB1
        best_key = max(
            self._key_to_action,
            key=lambda k: (
                self.q_values[k]
                + self.exploration_weight
                * np.sqrt(np.log(self._total_counts) / self.counts[k])
            ),
        )
        return self._key_to_action[best_key]

    def update(self, action, reward):
        key = self._action_key(action)
        self.counts[key] += 1
        self._total_counts += 1
        n = self.counts[key]
        self.q_values[key] += (reward - self.q_values[key]) / n

    def recommend(self):
        best_key = max(self._key_to_action, key=lambda k: self.q_values[k])
        return self._key_to_action[best_key]

    def fit(self, observational_data=None):
        """Warm-start the observational arm from data."""
        if observational_data is None:
            return

        reward_col = self.model.reward_variable
        obs_key = self._action_key({})

        for val in observational_data[reward_col]:
            if self.model.reward_mapping is not None:
                r = float(self.model.reward_mapping[val])
            else:
                r = float(val)
            self.counts[obs_key] += 1
            self._total_counts += 1
            n = self.counts[obs_key]
            self.q_values[obs_key] += (r - self.q_values[obs_key]) / n


class CausalThompsonSampling(CausalBanditPolicy):
    """
    Thompson Sampling policy for causal bandits.

    For binary rewards uses a Beta(alpha, beta) posterior per arm.
    For non-binary rewards (categorical with ``reward_mapping``, or
    user-specified ``reward_type="continuous"``) uses a Normal-Normal
    conjugate posterior.

    Parameters
    ----------
    model : pgmpy.bandits.CausalBanditModel
    """

    def __init__(self, model):
        super().__init__(model)
        self._use_beta = model.reward_type == "binary"

        if self._use_beta:
            self.alpha = {self._action_key(a): 1.0 for a in self.actions}
            self.beta = {self._action_key(a): 1.0 for a in self.actions}
        else:
            # Normal-Normal conjugate: track mu, tau (precision), counts
            self.mu = {self._action_key(a): 0.0 for a in self.actions}
            self.tau = {self._action_key(a): 1.0 for a in self.actions}
            self.counts = {self._action_key(a): 0 for a in self.actions}
            self._sum_rewards = {self._action_key(a): 0.0 for a in self.actions}

    def select_action(self):
        if self._use_beta:
            samples = {
                k: np.random.beta(self.alpha[k], self.beta[k])
                for k in self._key_to_action
            }
        else:
            samples = {
                k: np.random.normal(self.mu[k], 1.0 / np.sqrt(self.tau[k]))
                for k in self._key_to_action
            }
        best_key = max(samples, key=samples.get)
        return self._key_to_action[best_key]

    def update(self, action, reward):
        key = self._action_key(action)
        if self._use_beta:
            self.alpha[key] += reward
            self.beta[key] += 1.0 - reward
        else:
            self.counts[key] += 1
            self._sum_rewards[key] += reward
            n = self.counts[key]
            # Posterior precision and mean (unit prior variance, unit obs variance)
            self.tau[key] = 1.0 + n
            self.mu[key] = self._sum_rewards[key] / self.tau[key]

    def recommend(self):
        if self._use_beta:
            means = {
                k: self.alpha[k] / (self.alpha[k] + self.beta[k])
                for k in self._key_to_action
            }
        else:
            means = {k: self.mu[k] for k in self._key_to_action}
        best_key = max(means, key=means.get)
        return self._key_to_action[best_key]

    def fit(self, observational_data=None):
        """Warm-start the observational arm from data."""
        if observational_data is None:
            return

        reward_col = self.model.reward_variable
        obs_key = self._action_key({})

        for val in observational_data[reward_col]:
            if self.model.reward_mapping is not None:
                r = float(self.model.reward_mapping[val])
            else:
                r = float(val)
            self.update(self._key_to_action[obs_key], r)
