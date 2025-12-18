#!/usr/bin/env python
"""
HillClimbSearch - Score-based causal discovery using hill climbing optimization.

This module implements an sklearn-compatible version of the HillClimbSearch algorithm
for learning DAG structure from data.
"""
from collections import deque
from itertools import permutations
from typing import (
    Any,
    Callable,
    Deque,
    Generator,
    Hashable,
    List,
    Optional,
    Tuple,
    Union,
)

import networkx as nx
import pandas as pd
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseScoreCausalDiscovery
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method


class HillClimbSearch(_BaseScoreCausalDiscovery):
    """
    Score-based causal discovery using hill climbing optimization.

    This class implements the HillClimbSearch algorithm [1] for causal discovery.
    Given a tabular dataset, the algorithm estimates the causal structure among
    the variables in the data as a Directed Acyclic Graph (DAG). The algorithm
    works by iteratively making local modifications to the graph structure
    (adding, removing, or reversing edges) and keeping changes that improve
    the score until a local maximum is reached.

    The algorithm is a greedy local search method that:
    1. Starts from an initial graph (empty by default)
    2. Evaluates all possible single-edge modifications (add, delete, reverse)
    3. Applies the modification with the highest score improvement
    4. Repeats until no improvement can be made

    A tabu list is used to prevent the algorithm from immediately undoing recent
    changes, which helps avoid getting stuck in local optima.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. Also accepts a custom score instance that inherits
        from `StructureScore`.

    start_dag : DAG instance, default=None
        The starting point for the local search. By default, a completely
        disconnected network (no edges) is used. If provided, the DAG must
        contain exactly the same variables as in the data.

    tabu_length : int, default=100
        The number of recent graph modifications to store in the tabu list.
        These modifications cannot be reversed during the search procedure.
        This serves to enforce a wider exploration of the search space.

    max_indegree : int or None, default=None
        If provided, the procedure only searches among models where all nodes
        have at most `max_indegree` parents. This can significantly reduce
        the search space and computation time for large graphs.

    expert_knowledge : ExpertKnowledge instance, default=None
        Expert knowledge to be used with the algorithm. Expert knowledge
        allows specification of:

        - Required edges that must be present in the final graph
        - Forbidden edges that cannot be present in the final graph
        - Temporal ordering of nodes

    epsilon : float, default=1e-4
        Defines the exit condition. If the improvement in score is less
        than `epsilon`, the algorithm terminates and returns the learned model.

    max_iter : int, default=1e6
        The maximum number of iterations allowed. The algorithm terminates
        and returns the learned model when the number of iterations exceeds
        `max_iter`.

    use_cache : bool, default=True
        If True, uses caching of local scores for faster computation.
        Note: Caching only works for scoring methods which are decomposable.
        Can give incorrect results for custom non-decomposable scoring methods.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph as a DAG at a (local) score maximum.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model("alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the HillClimbSearch algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import HillClimbSearch
    >>> hc = HillClimbSearch(scoring_method="bic-d")
    >>> hc.fit(df)
    >>> hc.causal_graph_.edges()

    Use expert knowledge to constrain the search:

    >>> from pgmpy.estimators import ExpertKnowledge
    >>> expert = ExpertKnowledge(forbidden_edges=[("A", "B")])
    >>> hc = HillClimbSearch(scoring_method="bic-d", expert_knowledge=expert)
    >>> hc.fit(df)

    References
    ----------
    .. [1] Koller & Friedman, Probabilistic Graphical Models - Principles and
           Techniques, 2009, Section 18.4.3 (page 811ff)
    """

    def __init__(
        self,
        scoring_method: Optional[Union[str, StructureScore]] = None,
        start_dag: Optional[DAG] = None,
        tabu_length: int = 100,
        max_indegree: Optional[int] = None,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        epsilon: float = 1e-4,
        max_iter: int = int(1e6),
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.scoring_method = scoring_method
        self.start_dag = start_dag
        self.tabu_length = tabu_length
        self.max_indegree = max_indegree
        self.expert_knowledge = expert_knowledge
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.use_cache = use_cache
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame) -> "HillClimbSearch":
        """
        The fitting procedure for the HillClimbSearch algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : HillClimbSearch
            Returns the instance with the learned causal graph.
        """
        variables = list(X.columns)

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        score, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score

        # Step 1.2: Check the start_dag
        if self.start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(variables)
        elif not isinstance(self.start_dag, DAG) or not set(
            self.start_dag.nodes()
        ) == set(variables):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )
        else:
            start_dag = self.start_dag.copy()

        # Step 1.3: Check if expert knowledge was specified
        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            expert_knowledge = self.expert_knowledge

        # Step 1.3.1: If search_space in expert_knowledge is not None, limit the search space
        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(X.columns)

        # Step 1.4: Check if required edges cause a cycle
        start_dag.add_edges_from(expert_knowledge.required_edges)
        if not nx.is_directed_acyclic_graph(start_dag):
            raise ValueError(
                "required_edges create a cycle in start_dag. Please modify either required_edges or start_dag."
            )
        expert_knowledge._orient_temporal_forbidden_edges(start_dag, only_edges=False)
        start_dag.remove_edges_from(expert_knowledge.forbidden_edges)

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        max_indegree = self.max_indegree
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list: Deque[Tuple[str, Tuple[Hashable, Hashable]]] = deque(
            maxlen=self.tabu_length
        )
        current_model = start_dag

        if self.show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(self.max_iter))
        else:
            iteration = range(int(self.max_iter))

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        for _ in iteration:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    model=current_model,
                    variables=variables,
                    score=score_fn,
                    structure_score=score.structure_prior_ratio,
                    tabu_list=tabu_list,
                    max_indegree=max_indegree,
                    forbidden_edges=expert_knowledge.forbidden_edges,
                    required_edges=expert_knowledge.required_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )

            if best_operation is None or best_score_delta < self.epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X_node, Y_node = best_operation[1]
                current_model.remove_edge(X_node, Y_node)
                current_model.add_edge(Y_node, X_node)
                tabu_list.append(best_operation)

        # Step 3: Store results
        self.causal_graph_ = current_model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )

        return self

    @staticmethod
    def _legal_operations(
        model: DAG,
        variables: List[Hashable],
        score: Callable[[Any, List[Any]], float],
        structure_score: Callable[[str], float],
        tabu_list: Deque[Tuple[str, Tuple[Hashable, Hashable]]],
        max_indegree: int,
        forbidden_edges: List[Tuple[Hashable, Hashable]],
        required_edges: List[Tuple[Hashable, Hashable]],
    ) -> Generator[Tuple[Tuple[str, Tuple[Hashable, Hashable]], float], None, None]:
        """
        Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes.

        Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge.

        For details on scoring see Koller & Friedman, Probabilistic Graphical
        Models, Section 18.4.3.3 (page 818).

        If a number `max_indegree` is provided, only modifications that keep
        the number of parents for each node below `max_indegree` are considered.

        A list of edges can optionally be passed as `forbidden_edges` or
        `required_edges` to exclude those edges or to force them to be present
        in the model, respectively.

        Parameters
        ----------
        model : DAG
            The current graph model.

        variables : list
            List of all variable names in the dataset.

        score : callable
            The local scoring function.

        structure_score : callable
            The structure prior ratio function.

        tabu_list : deque
            List of recently performed operations that cannot be reversed.

        max_indegree : int
            Maximum number of parents allowed for any node.

        forbidden_edges : list
            Edges that are not allowed in the model.

        required_edges : list
            Edges that must be present in the model.

        Yields
        ------
        tuple
            A tuple of (operation, score_delta) where operation is a tuple of
            (operation_type, (source, target)) and score_delta is the change
            in score if this operation is applied.
        """
        tabu_set = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (operation not in tabu_set) and ((X, Y) not in forbidden_edges):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for X, Y in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_set) and ((X, Y) not in required_edges):
                old_parents = model.get_parents(Y)
                new_parents = [var for var in old_parents if var != X]
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_set) and ("flip", (Y, X)) not in tabu_set)
                    and ((X, Y) not in required_edges)
                    and ((Y, X) not in forbidden_edges)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = [var for var in old_Y_parents if var != X]
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)
