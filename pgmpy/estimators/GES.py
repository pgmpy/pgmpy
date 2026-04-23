import warnings
from collections.abc import Callable, Hashable, Iterable
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import PDAG
from pgmpy.estimators import (
    StructureEstimator,
    StructureScore,
)
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import get_scoring_method
from pgmpy.utils.mathext import powerset


class GES(StructureEstimator):
    """
    Implementation of Greedy Equivalence Search (GES) causal discovery / structure learning algorithm.

    GES is a score-based causal discovery / structure learning algorithm that works in three phases:
        1. Forward phase: New edges are added such that the model score improves.
        2. Backward phase: Edges are removed from the model such that the model score improves.
        3. Edge turning phase: Edge orientations are turned/flipped such that model score improves.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    [1] Chickering, David Maxwell. "Optimal structure identification with greedy search."
      Journal of machine learning research 3.Nov (2002): 507-554.

    [2] https://github.com/juangamella/ges

    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = False, **kwargs):
        warnings.warn(
            "GES is deprecated and will be removed in v1.3.0. Please use pgmpy.causal_discovery.GES instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.use_cache = use_cache

        super().__init__(data=data, **kwargs)

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

    def _legal_edge_additions(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all possible directed edge additions (u -> v) between non-adjacent nodes.
        """
        legal_edges: list[tuple[Hashable, Hashable]] = []

        for u, v in combinations(sorted(current_model.nodes()), 2):
            # Nodes must not be adjacent in any direction
            if not current_model.has_edge(u, v) and not current_model.has_edge(v, u):
                legal_edges.append((u, v))
                legal_edges.append((v, u))

        return legal_edges

    def _legal_edge_deletions(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all edges that can be considered for deletion.
        """
        return sorted(current_model.edges())

    def _legal_edge_turns(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all candidate edge turns (i.e., reverse directions of existing edges).
        """
        legal_turns: list[tuple[Hashable, Hashable]] = []

        for u, v in sorted(current_model.edges()):
            legal_turns.append((v, u))

        return legal_turns

    def insert(
        self,
        u: Any,
        v: Any,
        T: Iterable[Any],
        current_model: PDAG,
    ):
        """
        Perform insert(u -> v) with conditioning set T.
        """
        T = set(T)

        # Validity checks
        if current_model.has_edge(u, v) or current_model.has_edge(v, u):
            raise ValueError(f"Nodes u={u} and v={v} are already connected.")

        if T:
            if not T.issubset(current_model.undirected_neighbors(v)):
                raise ValueError(f"Not all nodes in T={T} are undirected neighbors of v={v}.")

            if current_model.all_neighbors(u) & T:
                raise ValueError(f"Some nodes in T={T} are adjacent to u={u}.")

        new_model = current_model.copy()

        # Add directed edge u -> v
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
    ):
        """
        Perform delete(u - v) or delete(u -> v) with conditioning set H.
        """
        na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)

        if not H.issubset(na_vu):
            raise ValueError(f"H={H} is not a subset of NA_vu={na_vu}.")

        new_model = current_model.copy()

        # Remove edge between u and v (both directions if present)
        new_model.remove_edges_from([(u, v), (v, u)])

        # Orient v - h as v -> h for all h in H
        for h in H:
            if new_model.has_undirected_edge(v, h):
                new_model.remove_edge(h, v)

        # For h in H ∩ Ne(u), orient u - h as u -> h
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
    ):
        """
        Perform turn operation (reverse or orient edge between u and v) with set C.
        """
        C = set(C)

        # Validity check (as per your current logic)
        if current_model.has_edge(u, v) and not current_model.has_edge(v, u):
            raise ValueError(f"The edge {u} -> {v} already exists.")

        new_model = current_model.copy()

        # Remove v -> u if present
        if new_model.has_edge(v, u):
            new_model.remove_edge(v, u)

        # Ensure u -> v exists
        if not new_model.has_edge(u, v):
            new_model.add_edge(u, v)

        # For each c in C: orient v - c as c -> v
        for c in C:
            if new_model.has_edge(v, c):
                new_model.remove_edge(v, c)
            new_model.add_edge(c, v)

        new_model.calibrate_directed_undirected_edges()
        return new_model

    def _score_valid_insertions(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid insert(u -> v) operations.
        """
        T0 = current_model.undirected_neighbors(v) - current_model.all_neighbors(u)

        power_set = powerset(list(T0))
        subsets = [[*T, False] for T in power_set]  # [elements..., passed_cond_2]
        valid_insert_ops = []

        while subsets:
            entry = subsets.pop(0)
            T, passed_cond_2 = set(entry[:-1]), entry[-1]

            na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
            na_vuT = na_vu.union(T)

            # Condition 1: NA_vu ∪ T is a clique
            cond_1 = current_model.is_clique(na_vuT)
            if not cond_1:
                # Prune supersets of T
                subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                continue

            # Condition 2: every semi-directed path from v to u intersects NA_vuT
            if passed_cond_2:
                cond_2 = True
            else:
                cond_2 = not current_model.has_semidirected_path(v, u, blocked_nodes=na_vuT)

                if cond_2:
                    # Mark supersets of T
                    for s in subsets:
                        if T.issubset(set(s[:-1])):
                            s[-1] = True

            if cond_1 and cond_2:
                parents_v = current_model.directed_parents(v)

                score_delta = score_fn(v, list(na_vuT | parents_v | {u})) - score_fn(v, list(na_vuT | parents_v))
                new_model = self.insert(u, v, T, current_model)
                if new_model.has_acyclic_extension():
                    valid_insert_ops.append((score_delta, u, v, T))

        return valid_insert_ops

    def _score_valid_deletions(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid delete(u - v) or delete(u -> v) operations.
        """
        if not current_model.has_edge(u, v):
            raise ValueError(f"No edge exists between nodes {u, v} to delete.")

        na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
        H0 = na_vu

        power_set = powerset(list(H0))
        subsets = [[*H, False] for H in power_set]  # [elements..., cond_1]
        valid_delete_ops = []

        while subsets:
            entry = subsets.pop(0)
            H, cond_1 = set(entry[:-1]), entry[-1]

            # Condition 1: NA_vu \ H is a clique
            if not cond_1 and current_model.is_clique(na_vu - H):
                cond_1 = True

                # Mark supersets of H
                for s in subsets:
                    if H.issubset(set(s[:-1])):
                        s[-1] = True

            if cond_1:
                aux = (na_vu - H) | current_model.directed_parents(v) | {u}

                score_delta = score_fn(v, list(aux - {u})) - score_fn(v, list(aux))

                valid_delete_ops.append((score_delta, u, v, H))

        return valid_delete_ops

    def _score_valid_turns(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ):
        """
        Dispatch turn operator depending on edge type.
        """
        if current_model.has_edge(u, v) and current_model.has_edge(v, u):
            return self._score_valid_turns_undirected(u, v, current_model, score_fn)
        else:
            return self._score_valid_turns_directed(u, v, current_model, score_fn)

    def _score_valid_turns_directed(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid turn(u -> v) operations.
        """
        T0 = current_model.undirected_neighbors(v) - current_model.all_neighbors(u)

        power_set = powerset(list(T0))
        subsets = [[*T, False] for T in power_set]  # [elements..., passed_cond_2]
        valid_turn_ops = []

        while subsets:
            entry = subsets.pop(0)
            T, passed_cond_2 = set(entry[:-1]), entry[-1]

            na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
            C = na_vu.union(T)

            # Condition 1: NA_vu ∪ T is a clique
            cond_1 = current_model.is_clique(C)
            if not cond_1:
                subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                continue

            # Condition 2: every semi-directed path from v to u intersects C ∪ Ne(u)
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

                new_score = score_fn(v, list(C | parents_v | {u})) + score_fn(u, list(parents_u - {v}))

                old_score = score_fn(v, list(C | parents_v)) + score_fn(u, list(parents_u))

                score_delta = new_score - old_score
                new_model = self.turn(u, v, T, current_model)
                if new_model.has_acyclic_extension():
                    valid_turn_ops.append((score_delta, u, v, T))

        return valid_turn_ops

    def _score_valid_turns_undirected(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid turn(u - v) operations.
        """
        non_adjacents = current_model.undirected_neighbors(v) - current_model.all_neighbors(u) - {u}

        if len(non_adjacents) == 0:
            return []

        C0 = current_model.undirected_neighbors(v) - {u}
        power_set = powerset(list(C0))

        # Only subsets containing at least one non-adjacent node
        subsets = [[*set(C), False] for C in power_set if len(set(C) & non_adjacents) > 0]

        valid_turn_ops = []

        while subsets:
            entry = subsets.pop(0)
            C = set(entry[:-1])

            # Condition 1: C is a clique
            cond_1 = current_model.is_clique(C)
            if not cond_1:
                subsets = [s for s in subsets if not C.issubset(set(s[:-1]))]
                continue

            subgraph = nx.DiGraph(current_model.subgraph(current_model.chain_component(v)))

            na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)

            # Separation condition
            if not self._separates({u, v}, C, na_vu - C, subgraph):
                continue

            parents_v = current_model.directed_parents(v)
            parents_u = current_model.directed_parents(u)

            new_score = score_fn(v, list(parents_v | C | {u})) + score_fn(u, list(parents_u | (C & na_vu)))

            old_score = score_fn(v, list(parents_v | C)) + score_fn(u, list(parents_u | (C & na_vu) | {v}))

            score_delta = new_score - old_score
            new_model = self.turn(u, v, C, current_model)
            if new_model.has_acyclic_extension():
                valid_turn_ops.append((score_delta, u, v, C))

        return valid_turn_ops

    def estimate(
        self,
        scoring_method: str | StructureScore | None = None,
        min_improvement: float = 1e-6,
        debug: bool = False,
    ) -> PDAG:
        """
        Estimates the DAG from the data.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
            be an instance of `StructureScore`.

        min_improvement: float
            The operation (edge addition, removal, or turning) would only be performed if the
            model score improves by atleast `min_improvement`.

        debug: bool
            Estimate the graph in debug mode, printing the corresponding increase in score at
            each step.

        Returns
        -------
        Estimated model: pgmpy.base.PDAG
            A `PDAG` at a (local) score maximum.

        Examples
        --------
        >>> import numpy as np
        >>> # Simulate some sample data from a known model to learn the model structure from
        >>> from pgmpy.utils import get_example_model
        >>> np.random.seed(42)
        >>> model = get_example_model("alarm")
        >>> model.seed = 42
        >>> df = model.simulate(int(1e3))

        >>> # Learn the model structure using GES algorithm from `df`
        >>> from pgmpy.estimators import GES
        >>> est = GES(df)
        >>> dag = est.estimate(scoring_method="bic-d")
        >>> len(dag.nodes())
        37
        >>> len(dag.edges())
        48
        """

        # Step 0: Initial checks and setup for arguments
        score_c: ScoreCache
        _, score_c = get_scoring_method(scoring_method, self.data, self.use_cache)
        score_fn = score_c.local_score

        # Step 1: Initialize an empty model.
        current_model = PDAG()  # if model is None else model
        current_model.add_nodes_from(list(self.data.columns))

        # Step 2: Forward step: Iteratively add edges till score stops improving.
        while True:
            potential_edges = self._legal_edge_additions(current_model)
            score_deltas = np.zeros(len(potential_edges))
            insertion_ops = []

            for index, (u, v) in enumerate(potential_edges):
                insertion_op = self._score_valid_insertions(u, v, current_model, score_fn)
                if insertion_op == []:
                    score_deltas[index] = 0
                    insertion_ops.append(None)
                else:
                    score_deltas[index] = max(insertion_op, key=lambda x: x[0])[0]
                    insertion_ops.append(max(insertion_op, key=lambda x: x[0]))

            if (len(potential_edges) == 0) or (np.all(score_deltas < min_improvement)):
                break

            edge_to_add = potential_edges[np.argmax(score_deltas)]
            op_to_add = insertion_ops[np.argmax(score_deltas)]

            current_model = self.insert(edge_to_add[0], edge_to_add[1], op_to_add[3], current_model)

            current_model = current_model.to_cpdag()

            if debug:
                print(f"Adding edge {edge_to_add[0]} -> {edge_to_add[1]}. Improves score by: {score_deltas.max()}")

        # Step 3: Backward Step: Iteratively remove edges till score stops improving.
        while True:
            potential_removals = self._legal_edge_deletions(current_model)

            score_deltas = np.zeros(len(potential_removals))
            deletion_ops = []

            for index, (u, v) in enumerate(potential_removals):
                deletion_op = self._score_valid_deletions(u, v, current_model, score_fn)
                if deletion_op == []:
                    score_deltas[index] = 0
                    deletion_ops.append(None)
                else:
                    score_deltas[index] = max(deletion_op, key=lambda x: x[0])[0]
                    deletion_ops.append(max(deletion_op, key=lambda x: x[0]))

            if (len(potential_removals) == 0) or (np.all(score_deltas < min_improvement)):
                break

            edge_to_remove = potential_removals[np.argmax(score_deltas)]
            op_to_delete = deletion_ops[np.argmax(score_deltas)]
            current_model = self.delete(edge_to_remove[0], edge_to_remove[1], op_to_delete[3], current_model)

            current_model = current_model.to_cpdag()

            if debug:
                print(
                    f"Removing edge {edge_to_remove[0]} -> {edge_to_remove[1]}. Improves score by: {score_deltas.max()}"
                )

        # Step 4: Turn Edges: Iteratively try to Turn edges till score stops improving.
        while True:
            potential_turns = self._legal_edge_turns(current_model)
            score_deltas = np.zeros(len(potential_turns))
            turn_ops = []

            for index, (u, v) in enumerate(potential_turns):
                turn_op = self._score_valid_turns(u, v, current_model, score_fn)
                if turn_op == []:
                    score_deltas[index] = 0
                    turn_ops.append(None)
                else:
                    score_deltas[index] = max(turn_op, key=lambda x: x[0])[0]
                    turn_ops.append(max(turn_op, key=lambda x: x[0]))

            if (len(potential_turns) == 0) or (np.all(score_deltas < min_improvement)):
                break

            edge_to_turn = potential_turns[np.argmax(score_deltas)]
            op_to_turn = turn_ops[np.argmax(score_deltas)]
            current_model = self.turn(edge_to_turn[0], edge_to_turn[1], op_to_turn[3], current_model)

            current_model = current_model.to_cpdag()

            if debug:
                print(f"Turning edge {edge_to_turn[0]} -> {edge_to_turn[1]}. Improves score by: {score_deltas.max()}")

        return current_model
