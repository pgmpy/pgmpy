#!/usr/bin/env python

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    pass


class CausalBanditMetrics:
    """
    Utility class for analyzing causal bandit performance.

    Provides methods for computing regret, causal effect estimation
    accuracy, and other relevant metrics.
    """

    @staticmethod
    def compute_regret(
        rewards: List[float],
        optimal_rewards: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute instantaneous and cumulative regret.

        Parameters
        ----------
        rewards : list of float
            Observed rewards
        optimal_rewards : list of float
            Optimal rewards from oracle

        Returns
        -------
        tuple of lists
            (instantaneous_regret, cumulative_regret)
        """
        regret = [opt - obs for opt, obs in zip(optimal_rewards, rewards)]
        cumulative_regret = np.cumsum(regret).tolist()

        return regret, cumulative_regret

    @staticmethod
    def compute_simple_regret(
        action_values: np.ndarray,
        optimal_action: int,
        selected_actions: List[int]
    ) -> List[float]:
        """
        Compute simple regret (regret of best action so far).

        Parameters
        ----------
        action_values : np.ndarray
            True action values
        optimal_action : int
            Index of optimal action
        selected_actions : list of int
            Actions selected at each round

        Returns
        -------
        list of float
            Simple regret over time
        """
        simple_regret = []
        best_so_far = 0

        for t, action in enumerate(selected_actions):
            if action_values[action] > action_values[best_so_far]:
                best_so_far = action

            regret = action_values[optimal_action] - action_values[best_so_far]
            simple_regret.append(regret)

        return simple_regret

    @staticmethod
    def plot_regret(
        regret_curves: Dict[str, List[float]],
        title: str = "Cumulative Regret",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot regret curves for multiple algorithms.

        Parameters
        ----------
        regret_curves : dict
            Algorithm name -> cumulative regret list
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        plt.figure(figsize=(10, 6))

        for algorithm, regret in regret_curves.items():
            plt.plot(regret, label=algorithm, linewidth=2)

        plt.xlabel("Round")
        plt.ylabel("Cumulative Regret")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_rewards(
        reward_curves: Dict[str, List[float]],
        window_size: int = 100,
        title: str = "Average Reward",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot smoothed reward curves.

        Parameters
        ----------
        reward_curves : dict
            Algorithm name -> reward list
        window_size : int
            Moving average window size
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        plt.figure(figsize=(10, 6))

        for algorithm, rewards in reward_curves.items():
            # Compute moving average
            smoothed = pd.Series(rewards).rolling(window_size).mean()
            plt.plot(smoothed, label=algorithm, linewidth=2)

        plt.xlabel("Round")
        plt.ylabel("Average Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def analyze_action_distribution(
        actions: List[int],
        action_names: Optional[List[str]] = None
    ) -> Dict[int, float]:
        """
        Analyze distribution of selected actions.

        Parameters
        ----------
        actions : list of int
            Selected actions over time
        action_names : list of str, optional
            Names for actions

        Returns
        -------
        dict
            Action index -> selection frequency
        """
        total = len(actions)
        if total == 0:
            return {}

        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Convert to frequencies
        action_freq = {action: count / total for action, count in action_counts.items()}

        return action_freq

    @staticmethod
    def compute_confidence_intervals(
        data: List[List[float]],
        confidence_level: float = 0.95
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute confidence intervals across multiple runs.

        Parameters
        ----------
        data : list of lists
            Multiple experimental runs
        confidence_level : float
            Confidence level for intervals

        Returns
        -------
        tuple of lists
            (mean, lower_bound, upper_bound)
        """
        data_array = np.array(data)
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)

        # Assuming normal distribution
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        margin = z_score * std / np.sqrt(len(data))

        lower_bound = mean - margin
        upper_bound = mean + margin

        return mean.tolist(), lower_bound.tolist(), upper_bound.tolist()

    @staticmethod
    def causal_effect_mse(
        estimated_effects: List[float],
        true_effects: List[float]
    ) -> float:
        """
        Compute MSE for causal effect estimation.

        Parameters
        ----------
        estimated_effects : list of float
            Estimated causal effects
        true_effects : list of float
            True causal effects

        Returns
        -------
        float
            Mean squared error
        """
        if len(estimated_effects) != len(true_effects):
            raise ValueError("Effect lists must have same length")

        mse = np.mean([(est - true) ** 2 for est, true in zip(estimated_effects, true_effects)])
        return float(mse)

    @staticmethod
    def intervention_efficiency(
        actions: List[int],
        outcome_variable_actions: List[int]
    ) -> float:
        """
        Compute efficiency of intervention selection.

        Parameters
        ----------
        actions : list of int
            Selected actions
        outcome_variable_actions : list of int
            Actions that directly affect outcome

        Returns
        -------
        float
            Fraction of actions that affect outcome
        """
        if not actions:
            return 0.0

        efficient_actions = sum(1 for action in actions if action in outcome_variable_actions)
        return efficient_actions / len(actions)


def create_synthetic_environment(
    n_variables: int = 5,
    n_actions: int = 3,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a synthetic causal bandit environment for testing.

    Parameters
    ----------
    n_variables : int
        Number of variables in the causal graph
    n_actions : int
        Number of actionable variables
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Environment configuration including graph, functions, etc.
    """
    if seed is not None:
        np.random.seed(seed)

    variables = [f"X{i}" for i in range(n_variables)]
    action_variables = variables[:n_actions]
    outcome_variable = variables[-1]

    # Generate random DAG
    from pgmpy.base import DAG
    graph = DAG()
    graph.add_nodes_from(variables)

    # Add random edges (ensuring acyclicity)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            if np.random.rand() < 0.3:  # 30% edge probability
                graph.add_edge(variables[i], variables[j])

    # Generate random causal coefficients
    causal_coefficients = {}
    for edge in graph.edges():
        causal_coefficients[edge] = np.random.normal(0, 0.5)

    return {
        "graph": graph,
        "variables": variables,
        "action_variables": action_variables,
        "outcome_variable": outcome_variable,
        "causal_coefficients": causal_coefficients,
    }


def run_bandit_comparison(
    environment: Dict[str, Any],
    policies: Dict[str, Any],
    n_rounds: int = 1000,
    n_runs: int = 10,
    seed: Optional[int] = None
) -> Dict[str, Dict[str, List]]:
    """
    Run comparison of multiple bandit policies.

    Parameters
    ----------
    environment : dict
        Causal bandit environment
    policies : dict
        Policy name -> policy configuration
    n_rounds : int
        Number of rounds per run
    n_runs : int
        Number of independent runs
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Results for each policy across all runs
    """
    if seed is not None:
        np.random.seed(seed)

    results = {}

    for policy_name, policy_config in policies.items():
        policy_results = {
            "rewards": [],
            "regrets": [],
            "actions": []
        }

        for run in range(n_runs):
            # Initialize policy and run
            # This would integrate with the actual policy classes
            run_rewards = np.random.randn(n_rounds).cumsum()  # Placeholder
            run_regrets = np.maximum(0, -run_rewards)  # Placeholder
            run_actions = np.random.randint(0, 3, n_rounds)  # Placeholder

            policy_results["rewards"].append(run_rewards.tolist())
            policy_results["regrets"].append(run_regrets.tolist())
            policy_results["actions"].append(run_actions.tolist())

        results[policy_name] = policy_results

    return results