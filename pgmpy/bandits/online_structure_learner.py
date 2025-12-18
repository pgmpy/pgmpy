#!/usr/bin/env python

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.base import DAG
from pgmpy.estimators import BaseEstimator
from pgmpy.estimators.StructureScore import BDeu, BIC


class OnlineCausalStructureLearner(BaseEstimator):
    """
    Online causal structure learning for bandit settings.

    Learns the causal graph structure incrementally as new data
    becomes available through bandit interactions.
    """

    def __init__(
        self,
        variables: List[str],
        action_variables: List[str],
        outcome_variable: str,
        initial_graph: Optional[DAG] = None,
        score_type: str = "bic",
        max_parents: int = 5,
        alpha: float = 0.05,
        update_frequency: int = 50,
    ):
        """
        Initialize online structure learner.

        Parameters
        ----------
        variables : list of str
            All variables in the system
        action_variables : list of str
            Variables that can be intervened upon
        outcome_variable : str
            The outcome variable of interest
        initial_graph : DAG, optional
            Initial graph structure hypothesis
        score_type : str
            Scoring method ("bic", "bdeu")
        max_parents : int
            Maximum number of parents per node
        alpha : float
            Significance level for independence tests
        update_frequency : int
            How often to update structure (in data points)
        """
        super().__init__()

        self.variables = variables
        self.action_variables = action_variables
        self.outcome_variable = outcome_variable
        self.score_type = score_type
        self.max_parents = max_parents
        self.alpha = alpha
        self.update_frequency = update_frequency

        # Initialize graph
        if initial_graph is not None:
            self.current_graph = initial_graph.copy()
            # Ensure all variables are in the graph
            for var in variables:
                if var not in self.current_graph.nodes():
                    self.current_graph.add_node(var)
        else:
            self.current_graph = DAG()
            self.current_graph.add_nodes_from(variables)

        # Data storage
        self.data_buffer = []
        self.update_count = 0
        self.total_observations = 0

        # Structure learning history
        self.structure_history = []
        self.score_history = []

    def add_observation(
        self,
        intervention: Dict[str, Any],
        outcome: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new observation from bandit interaction.

        Parameters
        ----------
        intervention : dict
            Intervention performed (variable: value mapping)
        outcome : float
            Observed outcome/reward
        context : dict, optional
            Additional observed variables
        """
        # Create observation record
        observation = {}

        # Add outcome
        observation[self.outcome_variable] = outcome

        # Add intervention variables
        for var in self.action_variables:
            observation[var] = intervention.get(var, 0)  # Default to 0 if not intervened

        # Add context variables
        if context:
            for var, value in context.items():
                if var in self.variables:
                    observation[var] = value

        # Fill missing variables with NaN
        for var in self.variables:
            if var not in observation:
                observation[var] = np.nan

        self.data_buffer.append(observation)
        self.total_observations += 1

        # Update structure if enough new data
        if len(self.data_buffer) >= self.update_frequency:
            self._update_structure()

    def _update_structure(self) -> None:
        """Update causal structure based on accumulated data."""
        if len(self.data_buffer) < 10:  # Need minimum data for structure learning
            return

        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer)

        # Remove rows with too many missing values
        df_clean = df.dropna(thresh=len(df.columns) * 0.7)

        if len(df_clean) < 5:  # Not enough clean data
            return

        # Perform structure search
        new_graph = self._structure_search(df_clean)

        if new_graph is not None:
            # Calculate improvement in score
            old_score = self._calculate_score(df_clean, self.current_graph)
            new_score = self._calculate_score(df_clean, new_graph)

            if new_score > old_score:
                self.current_graph = new_graph
                self.structure_history.append(new_graph.copy())
                self.score_history.append(new_score)

        # Clear buffer
        self.data_buffer = []
        self.update_count += 1

    def _structure_search(self, data: pd.DataFrame) -> Optional[DAG]:
        """
        Perform structure search on current data.

        Parameters
        ----------
        data : DataFrame
            Current observations

        Returns
        -------
        DAG or None
            Updated graph structure
        """
        try:
            # Simple greedy hill-climbing search
            current_graph = self.current_graph.copy()
            best_score = self._calculate_score(data, current_graph)
            improved = True

            while improved:
                improved = False
                best_operation = None

                # Try adding edges
                for parent in self.variables:
                    for child in self.variables:
                        if parent != child and not current_graph.has_edge(parent, child):
                            # Check if adding edge would create cycle
                            test_graph = current_graph.copy()
                            test_graph.add_edge(parent, child)

                            if nx.is_directed_acyclic_graph(test_graph):
                                # Check parent limit
                                if len(list(test_graph.predecessors(child))) <= self.max_parents:
                                    score = self._calculate_score(data, test_graph)
                                    if score > best_score:
                                        best_score = score
                                        best_operation = ("add", parent, child)
                                        improved = True

                # Try removing edges
                for parent, child in current_graph.edges():
                    test_graph = current_graph.copy()
                    test_graph.remove_edge(parent, child)

                    score = self._calculate_score(data, test_graph)
                    if score > best_score:
                        best_score = score
                        best_operation = ("remove", parent, child)
                        improved = True

                # Apply best operation
                if improved and best_operation:
                    operation, parent, child = best_operation
                    if operation == "add":
                        current_graph.add_edge(parent, child)
                    elif operation == "remove":
                        current_graph.remove_edge(parent, child)

            return current_graph

        except Exception as e:
            # Return None if structure search fails
            return None

    def _calculate_score(self, data: pd.DataFrame, graph: DAG) -> float:
        """
        Calculate structure score for given graph and data.

        Parameters
        ----------
        data : DataFrame
            Observations
        graph : DAG
            Graph structure to score

        Returns
        -------
        float
            Structure score (higher is better)
        """
        try:
            if self.score_type == "bic":
                scorer = BIC(data)
            elif self.score_type == "bdeu":
                scorer = BDeu(data)
            else:
                raise ValueError(f"Unknown score type: {self.score_type}")

            return scorer.score(graph)

        except Exception:
            # Return very negative score if scoring fails
            return -np.inf

    def get_current_graph(self) -> DAG:
        """Return current best estimate of causal structure."""
        return self.current_graph.copy()

    def get_causal_parents(self, variable: str) -> List[str]:
        """
        Get estimated causal parents of a variable.

        Parameters
        ----------
        variable : str
            Variable of interest

        Returns
        -------
        list of str
            Estimated causal parents
        """
        if variable not in self.current_graph.nodes():
            return []

        return list(self.current_graph.predecessors(variable))

    def get_causal_children(self, variable: str) -> List[str]:
        """
        Get estimated causal children of a variable.

        Parameters
        ----------
        variable : str
            Variable of interest

        Returns
        -------
        list of str
            Estimated causal children
        """
        if variable not in self.current_graph.nodes():
            return []

        return list(self.current_graph.successors(variable))

    def is_causal_path(self, from_var: str, to_var: str) -> bool:
        """
        Check if there's a causal path from one variable to another.

        Parameters
        ----------
        from_var : str
            Source variable
        to_var : str
            Target variable

        Returns
        -------
        bool
            True if causal path exists
        """
        try:
            return nx.has_path(self.current_graph, from_var, to_var)
        except nx.NodeNotFound:
            return False

    def estimate_intervention_targets(self, target_variable: str) -> List[str]:
        """
        Suggest intervention targets to affect a target variable.

        Parameters
        ----------
        target_variable : str
            Variable to influence

        Returns
        -------
        list of str
            Suggested intervention targets
        """
        targets = []

        # Find variables that have causal paths to target
        for action_var in self.action_variables:
            if self.is_causal_path(action_var, target_variable):
                targets.append(action_var)

        # Also include direct parents that are actionable
        parents = self.get_causal_parents(target_variable)
        for parent in parents:
            if parent in self.action_variables and parent not in targets:
                targets.append(parent)

        return targets

    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get summary of structure learning progress.

        Returns
        -------
        dict
            Summary statistics and current structure
        """
        return {
            "total_observations": self.total_observations,
            "structure_updates": self.update_count,
            "current_edges": list(self.current_graph.edges()),
            "current_nodes": list(self.current_graph.nodes()),
            "score_history": self.score_history.copy(),
            "n_edges": self.current_graph.number_of_edges(),
            "n_nodes": self.current_graph.number_of_nodes(),
        }