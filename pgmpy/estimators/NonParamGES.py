import warnings
from collections.abc import Hashable
from itertools import combinations
import logging

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from pgmpy import logger
from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery.ExpertKnowledge import ExpertKnowledge
from pgmpy.estimators import StructureEstimator, StructureScore


def conditional_log_likelihood(data: pd.DataFrame, node: str, parents: list[str], cv: int = 5) -> float:
    """
    Estimates the cross-validated conditional log-likelihood of a node given its parents
    using Kernel Density Estimation (KDE).
    
    This implements Component 1 of Nonparametric GES, where:
    P(X) = \\prod_j f_j(X_j | PA(j))
    L(X_j | PA_j) = L(X_j, PA_j) - L(PA_j)
    
    Parameters
    ----------
    data: pandas.DataFrame
        The data containing the node and its parents.
    node: str
        The node variable name.
    parents: list of str
        The parent variable names.
    cv: int
        Number of cross-validation folds for grid search.
        
    Returns
    -------
    float:
        The total cross-validated conditional log-likelihood.
    """
    # We use bandwidth search on a log scale
    params = {'bandwidth': np.logspace(-2, 1, 10)}
    
    vars_all = [node] + parents
    X_all = data[vars_all].values
    
    # Check if dimension is 1D -> sklearn needs 2D array
    if len(vars_all) == 1:
        X_all = X_all.reshape(-1, 1)
        
    grid_all = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=cv, n_jobs=-1)
    grid_all.fit(X_all)
    # The best_score_ is the mean score across folds, we multiply by cv 
    # to get an estimate of the total log-likelihood over the dataset
    ll_all = grid_all.best_score_ * cv
    
    if len(parents) == 0:
        return ll_all
        
    X_parents = data[parents].values
    if len(parents) == 1:
        X_parents = X_parents.reshape(-1, 1)
        
    grid_parents = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=cv, n_jobs=-1)
    grid_parents.fit(X_parents)
    ll_parents = grid_parents.best_score_ * cv
    
    return ll_all - ll_parents


def complexity_penalty(n_edges: int, n: int) -> float:
    """
    Returns the Le Cam-inspired complexity penalty for the graph.
    
    This implements Component 3 of Nonparametric GES:
    The penalty grows with graph complexity and is approximated as
    proportional to the number of edges scaled by log(n)/n.
    
    Parameters
    ----------
    n_edges: int
        Number of edges in the graph.
    n: int
        Number of samples.
        
    Returns
    -------
    float:
        The computed penalty.
    """
    return n_edges * np.log(n) / n


class NonParamGESScore(StructureScore):
    """
    Implements the pairwise test phi for Nonparametric GES.
    
    Component 4: Compares G vs G' with posterior odds ratio.
    """
    def __init__(self, data: pd.DataFrame, cv: int = 5, **kwargs):
        super().__init__(data, **kwargs)
        self.cv = cv
        
    def local_score(self, variable, parents):
        """
        Calculates local score. Not directly used inside NonParamGES test(G, G_prime) loop,
        but required to fulfill StructureScore interface.
        """
        n = self.data.shape[0]
        ll = conditional_log_likelihood(self.data, variable, parents, cv=self.cv)
        pen = complexity_penalty(len(parents), n)
        return ll - pen
        
    def test(self, X: pd.DataFrame, G: DAG, G_prime: DAG, lambda_threshold: float = 1.0) -> bool:
        """
        Pairwise test checking whether PR(G, G') > lambda_threshold.
        phi(X; G, G') = 1 if PR(G, G') > lambda else 0
        
        This satisfies decomposability by only comparing conditional log likelihood
        for nodes whose parents changed between G and G_prime.
        
        Returns True if G is preferred over G_prime.
        """
        changed_nodes = []
        for node in set(list(G.nodes()) + list(G_prime.nodes())):
            parents_G = set(G.predecessors(node)) if G.has_node(node) else set()
            parents_G_prime = set(G_prime.predecessors(node)) if G_prime.has_node(node) else set()
            if parents_G != parents_G_prime:
                changed_nodes.append(node)
                
        ll_G = 0.0
        ll_G_prime = 0.0
        
        for node in changed_nodes:
            pa_G = list(G.predecessors(node)) if G.has_node(node) else []
            pa_G_prime = list(G_prime.predecessors(node)) if G_prime.has_node(node) else []
            
            ll_G += conditional_log_likelihood(X, node, pa_G, cv=self.cv)
            ll_G_prime += conditional_log_likelihood(X, node, pa_G_prime, cv=self.cv)
            
        n = X.shape[0]
        pen_G = complexity_penalty(len(G.edges()), n)
        pen_G_prime = complexity_penalty(len(G_prime.edges()), n)
        
        log_PR = (ll_G - pen_G) - (ll_G_prime - pen_G_prime)
        
        return bool(log_PR > np.log(lambda_threshold))


class NonParamGES(StructureEstimator):
    """
    Implementation of Nonparametric Greedy Equivalence Search (GES) algorithm 
    with First-Accepted-Improvement logic.
    """
    def __init__(self, data: pd.DataFrame, cv: int = 5, lambda_threshold: float = 1.0, **kwargs):
        super().__init__(data=data, **kwargs)
        self.cv = cv
        self.score = NonParamGESScore(data, cv=cv)
        self.lambda_threshold = lambda_threshold
        
    def _legal_edge_additions(self, current_model: DAG, expert_knowledge: ExpertKnowledge):
        edges = []
        for u, v in combinations(current_model.nodes(), 2):
            if not (current_model.has_edge(u, v) or current_model.has_edge(v, u)):
                if not nx.has_path(current_model, v, u) and ((u, v) not in expert_knowledge.forbidden_edges):
                    edges.append((u, v))
                if not nx.has_path(current_model, u, v) and ((v, u) not in expert_knowledge.forbidden_edges):
                    edges.append((v, u))
        return edges

    def _legal_edge_removals(self, current_model: DAG, expert_knowledge: ExpertKnowledge):
        edges = []
        for u, v in current_model.edges():
            if (u, v) not in expert_knowledge.required_edges:
                edges.append((u, v))
        return edges

    def _legal_edge_flips(self, current_model: DAG, expert_knowledge: ExpertKnowledge):
        potential_flips = []
        edges = list(current_model.edges())
        for u, v in edges:
            if ((u, v) not in expert_knowledge.required_edges) and ((v, u) not in expert_knowledge.forbidden_edges):
                current_model.remove_edge(u, v)
                if not nx.has_path(current_model, u, v):
                    potential_flips.append((v, u))
                current_model.add_edge(u, v)
        return potential_flips

    def estimate(
        self,
        expert_knowledge: ExpertKnowledge | None = None,
        debug: bool = False,
    ) -> DAG:
        current_model = DAG()
        current_model.add_nodes_from(list(self.data.columns))
        
        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(self.data.columns)

        expert_knowledge._orient_temporal_forbidden_edges(current_model, only_edges=False)

        # Step 1: Forward step: Add edges until no additions are accepted
        while True:
            potential_edges = self._legal_edge_additions(current_model, expert_knowledge)
            
            accepted_improvement = False
            for u, v in potential_edges:
                candidate_model = current_model.copy()
                candidate_model.add_edge(u, v)
                
                # First-accepted-improvement
                if self.score.test(self.data, candidate_model, current_model, self.lambda_threshold):
                    current_model = candidate_model
                    accepted_improvement = True
                    if debug:
                        logger.info(f"Adding edge {u} -> {v}.")
                    break
                    
            if not accepted_improvement:
                break

        # Step 2: Backward Step: Remove edges until no removals are accepted
        while True:
            potential_removals = self._legal_edge_removals(current_model, expert_knowledge)
            
            accepted_improvement = False
            for u, v in potential_removals:
                candidate_model = current_model.copy()
                candidate_model.remove_edge(u, v)
                
                if self.score.test(self.data, candidate_model, current_model, self.lambda_threshold):
                    current_model = candidate_model
                    accepted_improvement = True
                    if debug:
                        logger.info(f"Removing edge {u} -> {v}.")
                    break
                    
            if not accepted_improvement:
                break

        # Step 3: Flip Edges: Try to flip edges until no flips are accepted
        while True:
            potential_flips = self._legal_edge_flips(current_model, expert_knowledge)
            
            accepted_improvement = False
            for new_u, new_v in potential_flips:
                candidate_model = current_model.copy()
                candidate_model.remove_edge(new_v, new_u)
                candidate_model.add_edge(new_u, new_v)
                
                if self.score.test(self.data, candidate_model, current_model, self.lambda_threshold):
                    current_model = candidate_model
                    accepted_improvement = True
                    if debug:
                        logger.info(f"Flipping edge {new_v} -> {new_u} to {new_u} -> {new_v}.")
                    break
                    
            if not accepted_improvement:
                break

        return current_model
