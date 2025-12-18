#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pgmpy.base import DAG
from pgmpy.inference.CausalInference import CausalInference


class CausalBanditModel:
    """
    Base class representing a causal bandit environment.

    This class manages the causal graph structure, intervention space,
    reward mechanism, and environment simulation for causal bandits.
    """

    def __init__(
        self,
        causal_graph: Union[DAG, str],
        action_variables: List[str],
        outcome_variable: str,
        confounders: Optional[List[str]] = None,
        latent_variables: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize causal bandit environment.

        Parameters
        ----------
        causal_graph : DAG or str
            The causal graph structure, either as a DAG object or path to file
        action_variables : list of str
            Variables that can be intervened upon (action space)
        outcome_variable : str
            The outcome/reward variable to optimize
        confounders : list of str, optional
            Known confounding variables
        latent_variables : list of str, optional
            Latent (unobserved) variables in the graph
        seed : int, optional
            Random seed for reproducibility
        """
        if isinstance(causal_graph, str):
            # Load DAG from file - placeholder for future file I/O
            raise NotImplementedError("Loading DAG from file not yet implemented")

        self.graph = causal_graph
        self.action_variables = action_variables
        self.outcome_variable = outcome_variable
        self.confounders = confounders or []
        self.latent_variables = latent_variables or []

        if seed is not None:
            np.random.seed(seed)

        # Initialize causal inference engine
        self.causal_inference = CausalInference(self.graph)

        # Validate graph structure
        self._validate_graph()

    def _validate_graph(self) -> None:
        """Validate that the causal graph contains required variables."""
        nodes = set(self.graph.nodes())

        if self.outcome_variable not in nodes:
            raise ValueError(f"Outcome variable {self.outcome_variable} not in graph")

        for action_var in self.action_variables:
            if action_var not in nodes:
                raise ValueError(f"Action variable {action_var} not in graph")

    def get_action_space(self) -> List[str]:
        """Return the available action variables."""
        return self.action_variables.copy()

    def get_valid_interventions(
        self, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get valid intervention sets given current context.

        Parameters
        ----------
        context : dict, optional
            Current context/state variables

        Returns
        -------
        list of dict
            Valid intervention dictionaries
        """
        # For now, return all possible single-variable interventions
        # Future versions can incorporate context-dependent constraints
        interventions = []

        # No intervention (observation)
        interventions.append({})

        # Single variable interventions with binary values
        for action_var in self.action_variables:
            interventions.append({action_var: 0})
            interventions.append({action_var: 1})

        return interventions

    def simulate_intervention(
        self, intervention: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Simulate the effect of an intervention and return reward.

        Parameters
        ----------
        intervention : dict
            Intervention to perform (variable: value mapping)
        context : dict, optional
            Current context/state variables

        Returns
        -------
        float
            Observed reward/outcome
        """
        # Use the causal graph to simulate intervention effects
        # This is a placeholder - real implementation would depend on
        # having a fully specified structural causal model

        # For demonstration, use a simple linear model with noise
        reward = 0.0

        # Base reward from context
        if context:
            reward += sum(context.values()) * 0.1

        # Intervention effects (placeholder - would be based on causal model)
        for var, value in intervention.items():
            if var in self.action_variables:
                reward += value * np.random.normal(0.5, 0.1)

        # Add noise
        reward += np.random.normal(0, 0.05)

        return reward

    def get_causal_effect(
        self, intervention: Dict[str, Any], estimand: Optional[str] = None
    ) -> float:
        """
        Estimate causal effect of intervention on outcome using pgmpy's inference.

        Parameters
        ----------
        intervention : dict
            Intervention to analyze (variable: value mapping)
        estimand : str, optional
            Specific causal estimand (default: ATE)

        Returns
        -------
        float
            Estimated causal effect
        """
        if not intervention:
            return 0.0

        try:
            # Use pgmpy's causal inference for proper identification and estimation
            # This requires a proper causal model with CPDs, so for now we use
            # a simplified approach based on causal paths in the graph

            effect = 0.0

            # For each intervention variable, check if it has a causal path to outcome
            for var, value in intervention.items():
                if var in self.action_variables:
                    # Check if there's a causal path from var to outcome
                    try:
                        import networkx as nx

                        if nx.has_path(self.graph, var, self.outcome_variable):
                            # Simple linear effect - in practice this would be estimated
                            # from the structural causal model or data
                            path_length = nx.shortest_path_length(
                                self.graph, var, self.outcome_variable
                            )
                            # Effect diminishes with path length
                            path_effect = value * (0.5 ** (path_length - 1))
                            effect += path_effect
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # No causal path - no effect
                        continue

            return effect

        except Exception:
            # Fallback to simple estimation if causal inference fails
            effect = 0.0
            for var, value in intervention.items():
                if var in self.action_variables:
                    effect += value * 0.3
            return effect


class CausalBanditPolicy(metaclass=ABCMeta):
    """
    Abstract base class for causal bandit policies.

    Defines the interface for exploration-exploitation strategies
    in causal bandit settings.
    """

    def __init__(self, action_space: List[Dict[str, Any]], **kwargs):
        """
        Initialize the policy.

        Parameters
        ----------
        action_space : list of dict
            Available actions/interventions
        **kwargs
            Policy-specific parameters
        """
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.t = 0  # Time step

    @abstractmethod
    def select_action(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Select an action given current context.

        Parameters
        ----------
        context : dict, optional
            Current context/state

        Returns
        -------
        int
            Index of selected action in action_space
        """
        pass

    @abstractmethod
    def update(
        self, action_idx: int, reward: float, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update policy based on observed reward.

        Parameters
        ----------
        action_idx : int
            Index of action taken
        reward : float
            Observed reward
        context : dict, optional
            Context when action was taken
        """
        pass

    def reset(self) -> None:
        """Reset policy to initial state."""
        self.t = 0


class CausalBanditLearner:
    """
    Coordinates the interaction between a causal bandit environment
    and a policy for online learning.
    """

    def __init__(
        self,
        environment: CausalBanditModel,
        policy: CausalBanditPolicy,
        horizon: int = 1000,
    ):
        """
        Initialize the learner.

        Parameters
        ----------
        environment : CausalBanditModel
            The causal bandit environment
        policy : CausalBanditPolicy
            The exploration policy to use
        horizon : int
            Number of rounds to run
        """
        self.environment = environment
        self.policy = policy
        self.horizon = horizon

        # Learning history
        self.history = {
            "actions": [],
            "rewards": [],
            "contexts": [],
            "regrets": [],
            "cumulative_regret": [],
        }

    def run(
        self,
        context_generator: Optional[callable] = None,
        oracle_policy: Optional[callable] = None,
    ) -> Dict[str, List]:
        """
        Run the bandit learning algorithm.

        Parameters
        ----------
        context_generator : callable, optional
            Function to generate contexts at each round
        oracle_policy : callable, optional
            Oracle policy for regret calculation

        Returns
        -------
        dict
            Learning history including actions, rewards, regrets
        """
        cumulative_regret = 0.0

        for t in range(self.horizon):
            # Generate context
            if context_generator:
                context = context_generator(t)
            else:
                context = {}

            # Select action
            action_idx = self.policy.select_action(context)
            intervention = self.environment.get_valid_interventions(context)[action_idx]

            # Observe reward
            reward = self.environment.simulate_intervention(intervention, context)

            # Calculate regret if oracle is available
            if oracle_policy:
                optimal_action = oracle_policy(context)
                optimal_reward = self.environment.simulate_intervention(
                    self.environment.get_valid_interventions(context)[optimal_action],
                    context,
                )
                regret = optimal_reward - reward
            else:
                regret = 0.0

            cumulative_regret += regret

            # Update policy
            self.policy.update(action_idx, reward, context)

            # Record history
            self.history["actions"].append(action_idx)
            self.history["rewards"].append(reward)
            self.history["contexts"].append(context)
            self.history["regrets"].append(regret)
            self.history["cumulative_regret"].append(cumulative_regret)

        return self.history

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics from learning history.

        Returns
        -------
        dict
            Performance metrics including cumulative regret, avg reward, etc.
        """
        if not self.history["rewards"]:
            return {}

        return {
            "cumulative_regret": (
                self.history["cumulative_regret"][-1]
                if self.history["cumulative_regret"]
                else 0.0
            ),
            "average_reward": np.mean(self.history["rewards"]),
            "total_reward": sum(self.history["rewards"]),
            "regret_per_round": (
                self.history["cumulative_regret"][-1] / len(self.history["rewards"])
                if self.history["cumulative_regret"]
                else 0.0
            ),
        }
