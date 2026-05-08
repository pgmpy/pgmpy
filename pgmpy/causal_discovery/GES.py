from collections.abc import Hashable, Iterable
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import PDAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from pgmpy.utils.mathext import powerset


class GES(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using Greedy Equivalence Search (GES).

    This class implements the GES algorithm [1]_ for causal discovery. Given a
    tabular dataset, the algorithm estimates the causal structure among the
    variables in the data as a Directed Acyclic Graph (DAG) or Partially
    Directed Acyclic Graph (PDAG).

    GES works in three phases:
        1. Forward phase: Edges are added to improve the model score.
        2. Backward phase: Edges are removed to improve the model score.
        3. Edge turning phase: Edge orientations are flipped to improve the score.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. Also accepts a custom score instance that inherits
        from `BaseStructureScore`.

    return_type : str, default='pdag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    min_improvement : float, default=1e-6
        The minimum score improvement required to perform an operation
        (edge addition, removal, or flipping). Operations with smaller
        improvements are not performed.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph at a (local) score maximum.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> import numpy as np
    >>> from pgmpy.example_models import load_model
    >>> np.random.seed(42)
    >>> model = load_model("bnlearn/alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the GES algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import GES
    >>> ges = GES(scoring_method="bic-d")
    >>> ges.fit(df)
    GES(scoring_method='bic-d')
    >>> ges.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.PDAG.PDAG object at 0x...>
    >>> ges.n_features_in_
    37

    References
    ----------
    .. [1] Chickering, David Maxwell. "Optimal structure identification with
           greedy search." Journal of machine learning research 3.Nov (2002):
           507-554.

    .. [2] https://github.com/juangamella/ges
    """

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        return_type: str = "pdag",
        min_improvement: float = 1e-6,
    ):
        self.scoring_method = scoring_method
        self.return_type = return_type
        self.min_improvement = min_improvement

    def _separates(
        self,
        S: set[Any],
        A: set[Any],
        B: set[Any],
        graph: nx.DiGraph,
    ) -> bool:
        """
        Check if S separates A and B in the graph.

        That is, every path from any node in A to any node in B
        intersects S.
        """
        for u in A:
            for v in B:
                for path in nx.all_simple_paths(graph, u, v):
                    if set(path).isdisjoint(S):
                        return False

        return True

    def _legal_edge_deletions(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all edges that can be considered for deletion.
        """
        return sorted(current_model.edges())

    def insert(
        self,
        u: Any,
        v: Any,
        T: Iterable[Any],
        current_model: PDAG,
    ) -> PDAG:
        """
        Perform insert(u -> v) with conditioning set T.
        """
        T = set(T)

        if current_model.has_edge(u, v) or current_model.has_edge(v, u):
            raise ValueError(f"Nodes u={u} and v={v} are already connected.")

        if T:
            if not T.issubset(current_model.undirected_neighbors(v)):
                raise ValueError(f"Not all nodes in T={T} are undirected neighbors of v={v}.")

            if current_model.all_neighbors(u) & T:
                raise ValueError(f"Some nodes in T={T} are adjacent to u={u}.")

        new_model = current_model.copy()
        new_model.add_edge(u, v)

        # Orient v - t as t -> v for all t in T
        remove_edges = [(v, t) for t in T]
        new_model.remove_edges_from(remove_edges)

        new_model.calibrate_directed_undirected_edges()
        return new_model

    def delete(
        self,
        u: Any,
        v: Any,
        H: set[Any],
        current_model: PDAG,
    ) -> PDAG:
        """
        Perform delete(u - v) or delete(u -> v) with conditioning set H.
        """
        na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)

        if not H.issubset(na_vu):
            raise ValueError(f"H={H} is not a subset of NA_vu={na_vu}.")

        new_model = current_model.copy()
        new_model.remove_edges_from([(u, v), (v, u)])

        for h in H:
            if new_model.has_undirected_edge(v, h):
                new_model.remove_edge(h, v)

        u_neighbors = set(new_model.undirected_neighbors(u))
        for h in H & u_neighbors:
            new_model.remove_edge(h, u)

        new_model.calibrate_directed_undirected_edges()
        return new_model

    def turn(
        self,
        u: Any,
        v: Any,
        C: Iterable[Any],
        current_model: PDAG,
    ) -> PDAG:
        """
        Perform turn operation (reverse or orient edge between u and v) with set C.
        """
        C = set(C)

        if current_model.has_edge(u, v) and not current_model.has_edge(v, u):
            raise ValueError(f"The edge {u} -> {v} already exists.")

        new_model = current_model.copy()

        if new_model.has_edge(v, u):
            new_model.remove_edge(v, u)

        if not new_model.has_edge(u, v):
            new_model.add_edge(u, v)

        for c in C:
            if new_model.has_edge(v, c):
                new_model.remove_edge(v, c)
            new_model.add_edge(c, v)

        new_model.calibrate_directed_undirected_edges()
        return new_model

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the GES algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.GES
            Returns the instance with the fitted attributes.
        """
        self.variables_ = list(X.columns)

        def ordered_tuple(nodes: Iterable[Any], model: PDAG) -> tuple[Any, ...]:
            node_set = set(nodes)
            return tuple(node for node in model.nodes() if node in node_set)

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check score
        score = get_scoring_method(self.scoring_method, X)
        score_fn = score.local_score

        # Step 1.2: Initialize the starting PDAG
        current_model = PDAG()
        current_model.add_nodes_from(self.variables_)

        # Step 2: Forward phase. Iteratively add edges till score stops improving.
        while True:
            potential_edges = []
            for u, v in combinations(sorted(current_model.nodes()), 2):
                if not current_model.has_edge(u, v) and not current_model.has_edge(v, u):
                    potential_edges.append((u, v))
                    potential_edges.append((v, u))

            score_deltas = np.zeros(len(potential_edges))
            insertion_ops: list[tuple[float, Any, Any, set[Any]] | None] = []

            for index, (u, v) in enumerate(potential_edges):
                T0 = current_model.undirected_neighbors(v) - current_model.all_neighbors(u)
                subsets = [[*T, False] for T in powerset(list(T0))]
                valid_insert_ops = []

                while subsets:
                    entry = subsets.pop(0)
                    T, passed_cond_2 = set(entry[:-1]), entry[-1]

                    na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
                    na_vuT = na_vu.union(T)

                    cond_1 = current_model.is_clique(na_vuT)
                    if not cond_1:
                        subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                        continue

                    if passed_cond_2:
                        cond_2 = True
                    else:
                        cond_2 = not current_model.has_semidirected_path(v, u, blocked_nodes=na_vuT)

                        if cond_2:
                            for s in subsets:
                                if T.issubset(set(s[:-1])):
                                    s[-1] = True

                    if cond_1 and cond_2:
                        parents_v = current_model.directed_parents(v)
                        new_parents = ordered_tuple(na_vuT | parents_v | {u}, current_model)
                        old_parents = ordered_tuple(na_vuT | parents_v, current_model)
                        score_delta = score_fn(v, new_parents) - score_fn(v, old_parents)

                        new_model = self.insert(u, v, T, current_model)
                        if new_model.has_acyclic_extension():
                            valid_insert_ops.append((score_delta, u, v, T))

                if valid_insert_ops == []:
                    score_deltas[index] = 0
                    insertion_ops.append(None)
                else:
                    best_op = max(valid_insert_ops, key=lambda x: x[0])
                    score_deltas[index] = best_op[0]
                    insertion_ops.append(best_op)

            if (len(potential_edges) == 0) or (np.all(score_deltas < self.min_improvement)):
                break

            op_to_add = insertion_ops[np.argmax(score_deltas)]
            if op_to_add is None:
                break

            current_model = self.insert(op_to_add[1], op_to_add[2], op_to_add[3], current_model)
            current_model = current_model.to_cpdag()

        # Step 3: Backward phase. Iteratively remove edges till score stops improving.
        while True:
            potential_removals = self._legal_edge_deletions(current_model)
            score_deltas = np.zeros(len(potential_removals))
            deletion_ops: list[tuple[float, Any, Any, set[Any]] | None] = []

            for index, (u, v) in enumerate(potential_removals):
                if not current_model.has_edge(u, v):
                    raise ValueError(f"No edge exists between nodes {(u, v)} to delete.")

                na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
                subsets = [[*H, False] for H in powerset(list(na_vu))]
                valid_delete_ops = []

                while subsets:
                    entry = subsets.pop(0)
                    H, cond_1 = set(entry[:-1]), entry[-1]

                    if not cond_1 and current_model.is_clique(na_vu - H):
                        cond_1 = True
                        for s in subsets:
                            if H.issubset(set(s[:-1])):
                                s[-1] = True

                    if cond_1:
                        aux = (na_vu - H) | current_model.directed_parents(v) | {u}
                        old_parents = ordered_tuple(aux, current_model)
                        new_parents = ordered_tuple(aux - {u}, current_model)
                        score_delta = score_fn(v, new_parents) - score_fn(v, old_parents)
                        valid_delete_ops.append((score_delta, u, v, H))

                if valid_delete_ops == []:
                    score_deltas[index] = 0
                    deletion_ops.append(None)
                else:
                    best_op = max(valid_delete_ops, key=lambda x: x[0])
                    score_deltas[index] = best_op[0]
                    deletion_ops.append(best_op)

            if (len(potential_removals) == 0) or (np.all(score_deltas < self.min_improvement)):
                break

            op_to_delete = deletion_ops[np.argmax(score_deltas)]
            if op_to_delete is None:
                break

            current_model = self.delete(op_to_delete[1], op_to_delete[2], op_to_delete[3], current_model)
            current_model = current_model.to_cpdag()

        # Step 4: Turning phase. Iteratively reorient edges till score stops improving.
        while True:
            potential_turns = []
            for u, v in sorted(current_model.edges()):
                potential_turns.append((v, u))

            score_deltas = np.zeros(len(potential_turns))
            turn_ops: list[tuple[float, Any, Any, set[Any]] | None] = []

            for index, (u, v) in enumerate(potential_turns):
                valid_turn_ops = []

                if current_model.has_edge(u, v) and current_model.has_edge(v, u):
                    non_adjacents = current_model.undirected_neighbors(v) - current_model.all_neighbors(u) - {u}

                    if len(non_adjacents) > 0:
                        C0 = current_model.undirected_neighbors(v) - {u}
                        subsets = [[*set(C), False] for C in powerset(list(C0)) if len(set(C) & non_adjacents) > 0]

                        while subsets:
                            entry = subsets.pop(0)
                            C = set(entry[:-1])

                            cond_1 = current_model.is_clique(C)
                            if not cond_1:
                                subsets = [s for s in subsets if not C.issubset(set(s[:-1]))]
                                continue

                            subgraph = nx.DiGraph(current_model.subgraph(current_model.chain_component(v)))
                            na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)

                            if not self._separates({u, v}, C, na_vu - C, subgraph):
                                continue

                            parents_v = current_model.directed_parents(v)
                            parents_u = current_model.directed_parents(u)

                            new_score = score_fn(v, ordered_tuple(parents_v | C | {u}, current_model)) + score_fn(
                                u, ordered_tuple(parents_u | (C & na_vu), current_model)
                            )
                            old_score = score_fn(v, ordered_tuple(parents_v | C, current_model)) + score_fn(
                                u, ordered_tuple(parents_u | (C & na_vu) | {v}, current_model)
                            )
                            score_delta = new_score - old_score

                            new_model = self.turn(u, v, C, current_model)
                            if new_model.has_acyclic_extension():
                                valid_turn_ops.append((score_delta, u, v, C))
                else:
                    T0 = current_model.undirected_neighbors(v) - current_model.all_neighbors(u)
                    subsets = [[*T, False] for T in powerset(list(T0))]

                    while subsets:
                        entry = subsets.pop(0)
                        T, passed_cond_2 = set(entry[:-1]), entry[-1]

                        na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
                        C = na_vu.union(T)

                        cond_1 = current_model.is_clique(C)
                        if not cond_1:
                            subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                            continue

                        if passed_cond_2:
                            cond_2 = True
                        else:
                            cond_2 = not current_model.has_semidirected_path(
                                v,
                                u,
                                blocked_nodes=C | current_model.undirected_neighbors(u),
                                ignore_direct_edge=True,
                            )

                            if cond_2:
                                for s in subsets:
                                    if T.issubset(set(s[:-1])):
                                        s[-1] = True

                        if cond_1 and cond_2:
                            parents_v = current_model.directed_parents(v)
                            parents_u = current_model.directed_parents(u)

                            new_score = score_fn(v, ordered_tuple(C | parents_v | {u}, current_model)) + score_fn(
                                u, ordered_tuple(parents_u - {v}, current_model)
                            )
                            old_score = score_fn(v, ordered_tuple(C | parents_v, current_model)) + score_fn(
                                u, ordered_tuple(parents_u, current_model)
                            )
                            score_delta = new_score - old_score

                            new_model = self.turn(u, v, T, current_model)
                            if new_model.has_acyclic_extension():
                                valid_turn_ops.append((score_delta, u, v, T))

                if valid_turn_ops == []:
                    score_deltas[index] = 0
                    turn_ops.append(None)
                else:
                    best_op = max(valid_turn_ops, key=lambda x: x[0])
                    score_deltas[index] = best_op[0]
                    turn_ops.append(best_op)

            if (len(potential_turns) == 0) or (np.all(score_deltas < self.min_improvement)):
                break

            op_to_turn = turn_ops[np.argmax(score_deltas)]
            if op_to_turn is None:
                break

            current_model = self.turn(op_to_turn[1], op_to_turn[2], op_to_turn[3], current_model)
            current_model = current_model.to_cpdag()

        # Step 5: Store results
        current_model = current_model.to_cpdag()

        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model.to_dag()
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model
        else:
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self
