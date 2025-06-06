from itertools import chain, combinations

import networkx as nx
import numpy as np

from pgmpy import config
from pgmpy.base import DAG, PDAG
from pgmpy.estimators import (
    AIC,
    BIC,
    K2,
    AICCondGauss,
    AICGauss,
    BDeu,
    BDs,
    BICCondGauss,
    BICGauss,
    ExpertKnowledge,
    LogLikelihoodCondGauss,
    LogLikelihoodGauss,
    StructureEstimator,
    StructureScore,
    get_scoring_method,
)
from pgmpy.global_vars import logger


class GES(StructureEstimator):
    """
    Implementation of Greedy Equivalence Search (GES) causal discovery / structure learning algorithm.

    GES is a score-based causal discovery / structure learning algorithm that works in three phases:
        1. Forward phase: New edges are added such that the model score improves.
        2. Backward phase: Edges are removed from the model such that the model score improves.
        3. Edge flipping phase: Edge orientations are flipped such that model score improves.

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
    Chickering, David Maxwell. "Optimal structure identification with greedy search." Journal of machine learning research 3.Nov (2002): 507-554.
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache

        super(GES, self).__init__(data=data, **kwargs)

    def _legal_edge_additions(self, current_model, expert_knowledge):
        """
        Returns a list of all edges that can be added to the graph (between non adjacent nodes).
        """
        legal_edges = []
        for u, v in combinations(current_model.nodes(), 2):
            if (not current_model.has_edge(u, v)) and (
                not current_model.has_edge(v, u)
            ):
                legal_edges.append((u, v))
                legal_edges.append((v, u))
        return legal_edges

    def _legal_edge_deletions(self, current_model, expert_knowledge):
        directed_edges = [
            (u, v) for u, v in current_model.edges() if not current_model.has_edge(v, u)
        ]

        undirected_edges = [
            (u, v)
            for u, v in current_model.edges()
            if current_model.has_edge(v, u) and u > v
        ]

        legal_edges = directed_edges + undirected_edges
        return legal_edges

    def _legal_edge_turns(self, current_model, expert_knowledge):
        return list(current_model.edges)

    @staticmethod
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _score_valid_insertions(self, u, v, current_model, score_fn):
        T0 = current_model.undirected_neighbors(v) - current_model.adjacent_neighbors(u)

        power_set = self.powerset(T0)
        subsets = [[*T, False] for T in power_set]
        valid_insert_ops = []

        while len(subsets) > 0:
            entry = subsets.pop(0)
            T, passed_cond_2 = set(entry[:-1]), entry[-1]

            na_vu = current_model.undirected_neighbors(
                v
            ) & current_model.adjacent_neighbors(u)
            na_vuT = na_vu.union(T)

            # === Condition 1: na_yx ∪ T is a clique
            cond_1 = current_model.is_clique(na_vuT)
            if not cond_1:
                # Prune supersets of T
                subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                continue

            # === Condition 2: every semi-directed path from y to x intersects na_yxT
            if passed_cond_2:
                cond_2 = True
            else:
                cond_2 = True
                for path in nx.all_simple_paths(current_model, v, u):
                    if not na_vuT.intersection(set(path)):
                        cond_2 = False
                        break
                if cond_2:
                    # Mark supersets of T
                    for s in subsets:
                        if T.issubset(set(s[:-1])):
                            s[-1] = True

            if cond_1 and cond_2:
                new_model = self.insert(u, v, T, current_model)
                parents_v = current_model.directed_parents(v)
                # print(na_vuT)
                # print(parents_v)
                # print(na_vuT.union(parents_v).union(u))
                score_delta = score_fn(
                    v, list(na_vuT.union(parents_v).union({u}))
                ) - score_fn(v, list(na_vuT.union(parents_v)))

                valid_insert_ops.append((score_delta, new_model, u, v, T))

        return valid_insert_ops

    def insert(self, u, v, T, current_model):
        T = sorted(T)
        # print("\nInsert - ",u,v,T)
        # print('3.\n', current_model.edges)
        # if current_model[x, y] != 0 or current_model[y, x] != 0:
        #    raise ValueError("x=%d and y=%d are already connected" % (x, y))
        # print('insert - ', current_model.edges, current_model.undirected_neighbors('Survived'))
        if len(T) == 0:
            pass
        elif not (set(T) <= current_model.undirected_neighbors(v)):
            raise ValueError("Not all nodes in T=%s are neighbors of y=%s" % (T, v))
        elif len(current_model.all_neighbors(u) & set(T)) != 0:
            raise ValueError("Some nodes in T=%s are adjacent to x=%s" % (T, u))
        # Apply operator
        new_model = current_model.copy()
        # print('4.\n', new_model.edges)
        # Add edge x -> y
        new_edges = [(u, v)]
        remove_edges = []
        # Orient edges t - v to t -> v, for t in T
        for node in T:
            remove_edges.append((v, node))
        new_model.add_edges_from(new_edges)
        new_model.remove_edges_from(remove_edges)
        new_model.add_nodes_from(current_model.nodes)
        # print(u, v, T)
        return new_model

    def _score_valid_deletions(self, u, v, current_model, score_fn):
        # Check inputs
        if not current_model.has_edge(u, v):
            print(current_model.edges)
            raise ValueError(f"There is no (un)directed edge from to delete {u,v}")
        # One-hot encode all subsets of H0, plus one column to mark if
        na_vu = current_model.undirected_neighbors(
            v
        ) & current_model.adjacent_neighbors(u)
        H0 = sorted(na_vu)

        power_set = self.powerset(H0)
        subsets = [[*H, False] for H in power_set]
        valid_delete_ops = []

        while len(subsets) > 0:
            # Access the next subset
            entry = subsets.pop(0)
            H, cond_1 = set(entry[:-1]), entry[-1]
            subsets = subsets[1:]
            # Check if the validity condition holds for H, i.e. that
            # NA_yx \ H is a clique.
            # If it has not been tested previously for a subset of H,
            # check it now
            if not cond_1 and current_model.is_clique(na_vu - set(H)):
                cond_1 = True
                # For all supersets H' of H, the validity condition will also hold

                # Mark all supersets of H
                for s in subsets:
                    if H.issubset(set(s[:-1])):
                        s[-1] = True
            # If the validity condition holds, apply operator and compute its score

            if cond_1:
                # Apply operator
                new_model = self.delete(u, v, H, current_model)
                # Compute the change in score
                aux = (na_vu - set(H)) | current_model.directed_parents(v)
                # print(x,y,H,"na_yx:",na_yx,"old:",aux,"new:", aux - {x})

                score_delta = score_fn(
                    v, (na_vu - set(H)).union(current_model.directed_parents(v))
                ) - score_fn(
                    v,
                    (na_vu - set(H))
                    .union(current_model.directed_parents(v))
                    .union({u}),
                )

                # Add to the list of valid operators
                valid_delete_ops.append((score_delta, new_model, u, v, H))

        # Return all the valid operators
        return valid_delete_ops

    def delete(self, u, v, H, current_model):
        na_vu = current_model.undirected_neighbors(
            v
        ) & current_model.adjacent_neighbors(u)
        if not H <= na_vu:
            raise ValueError(
                "The given set H is not valid, H=%s is not a subset of NA_yx=%s"
                % (H, na_vu)
            )

        new_model = current_model.copy()
        # delete the edge between x and y
        new_model.remove_edges_from([(u, v), (v, u)])
        # orient the undirected edges between y and H towards H
        for h in sorted(H):
            if new_model.has_undirected_edge(v, h):
                new_model.remove_edge(v, h)
                new_model.remove_edge(h, v)

        # For any h in both H and neighbors of x, remove undirected edge x-h
        x_h = sorted(H & set(new_model.undirected_neighbors(u)))
        for h in x_h:
            new_model.remove_edge(u, h)
            new_model.remove_edge(h, u)

        return new_model

    def _score_valid_turns(self, u, v, current_model, score_fn):
        if current_model.has_edge(u, v) and current_model.has_edge(v, u):
            return self._score_valid_turn_undir(u, v, current_model, score_fn)
        else:
            return self._score_valid_turn_dir(u, v, current_model, score_fn)

    def _score_valid_turn_dir(self, u, v, current_model, score_fn):
        T0 = current_model.undirected_neighbors(v) - current_model.adjacent_neighbors(u)

        power_set = self.powerset(T0)
        subsets = [[*T, False] for T in power_set]
        valid_turn_ops = []

        while len(subsets) > 0:
            entry = subsets.pop(0)
            T, passed_cond_2 = set(entry[:-1]), entry[-1]

            C = current_model.undirected_neighbors(
                v
            ) & current_model.adjacent_neighbors(u)
            C = C.union(T)

            # === Condition 1: na_yx ∪ T is a clique
            cond_1 = current_model.is_clique(C)
            if not cond_1:
                # Prune supersets of T
                subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                continue

            # === Condition 2: every semi-directed path from y to x intersects na_yxT
            if passed_cond_2:
                cond_2 = True
            else:
                cond_2 = True
                for path in nx.all_simple_paths(current_model, v, u):
                    if path == [v, u]:
                        pass
                    elif (
                        len(C | current_model.undirected_neighbors(u) & set(path)) == 0
                    ):
                        cond_2 = False
                        break
                if cond_2:
                    # Mark supersets of T
                    for s in subsets:
                        if T.issubset(set(s[:-1])):
                            s[-1] = True

            if cond_1 and cond_2:
                new_model = self.turn(u, v, T, current_model)
                parents_v = current_model.directed_parents(v)
                # print(na_vuT)
                # print(parents_v)
                # print(na_vuT.union(parents_v).union(u))
                print(v)
                new_score = score_fn(v, list(C.union(parents_v).union({u}))) + score_fn(
                    u, list(current_model.directed_parents(u) - {v})
                )

                old_score = score_fn(v, list(C.union(parents_v))) + score_fn(
                    u, list(current_model.directed_parents(u))
                )

                score_delta = new_score - old_score
                valid_turn_ops.append((score_delta, new_model, u, v, T))

        return valid_turn_ops

    def _score_valid_turn_undir(self, u, v, current_model, score_fn):
        non_adjacents = list(
            current_model.undirected_neigbors(v)
            - current_model.adjacent_neigbors(u)
            - {u}
        )
        if len(non_adjacents) == 0:
            logger.INFO("    turn(%d,%d) : ne(y) \\ adj(x) = Ø => stopping" % (u, v))
            return []

        C0 = sorted(current_model.undirected_neigbors(v) - {u})
        power_set = self.powerset(C0)
        valid_subsets = []

        non_adjacents = [
            n
            for n in C0
            if not current_model.has_edge(u, n) and not current_model.has_edge(n, u)
        ]
        for i in range(1, 2 ** len(C0)):
            subset = {C0[j] for j in range(len(C0)) if (i >> j) & 1}
            # Keep only subsets containing at least one non-adjacent node to x
            if any(node in subset for node in non_adjacents):
                valid_subsets.append(subset)

        valid_operators = []
        while len(valid_subsets) > 0:
            C = subsets.pop(0)
            cond_1 = current_model.is_clique(C)
            if not cond_1:
                # Remove from consideration all other sets C' which
                # contain C, as the clique condition will also not hold
                subsets = [S for j, S in enumerate(subsets) if not (C < S or C == S)]

            subgraph = current_model.induced_subgraph(current_model.chain_component(v))
            na_vu = current_model.undirected_neighbors(
                v
            ) & current_model.adjacent_neighbors(u)

            if not current_model.separates({u, v}, C, na_vu - C, subgraph):
                continue

            new_model = self.turn(u, v, C, current_model)

            new_score = score_fn(
                v, current_model.directed_parents(v) | C | {u}
            ) + score_fn(u, current_model.directed_parents(u) | (C & na_vu))
            old_score = score_fn(v, current_model.directed_parents(v) | C) + score_fn(
                u, current_model.directed_parents(u) | (C & na_vu) | {v}
            )

            score_delta = new_score - old_score
            valid_operators.append((score_delta, current_model, u, v, C))

        return valid_operators

    def turn(self, u, v, C, current_model):

        if current_model.has_edge(v, u):
            current_model.remove_edges_from([(v, u)])
        current_model.add_edges_from([(u, v)])
        for c in C:
            if current_model.has_edge(v, c):
                current_model.remove_edges_from([(v, c)])
            current_model.add_edges_from([(c, v)])
        return current_model

    def _legal_edge_removals(self, current_model, expert_knowledge):
        """
        Returns a list of all edges that can be removed from the graph such that it remains a DAG.
        """
        edges = set()
        for u, v in current_model.edges():
            if (v, u) not in current_model.edges:
                edges.add((u, v))
            else:
                undir_edge = (u, v) if u > v else (v, u)
                edges.add(undir_edge)

        return list(edges)

    def _is_dag(self, current_model):
        for u, v in current_model.edges:
            if (v, u) in current_model.edges:
                return False
        return True

    def estimate(
        self,
        scoring_method="bic-d",
        expert_knowledge=None,
        min_improvement=1e-6,
        debug=False,
    ):
        """
        Estimates the DAG from the data.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
            be an instance of `StructureScore`.

        expert_knowledge: pgmpy.estimators.ExpertKnowledge instance (default: None)
            Expert knowledge to be used with the algorithm. Expert knowledge
            allows specification of required and forbidden edges, as well as temporal
            order of nodes.

        min_improvement: float
            The operation (edge addition, removal, or flipping) would only be performed if the
            model score improves by atleast `min_improvement`.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> # Simulate some sample data from a known model to learn the model structure from
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model('alarm')
        >>> df = model.simulate(int(1e3))

        >>> # Learn the model structure using GES algorithm from `df`
        >>> from pgmpy.estimators import GES
        >>> est = GES(data)
        >>> dag = est.estimate(scoring_method='bic-d')
        >>> len(dag.nodes())
        37
        >>> len(dag.edges())
        45
        """

        # Step 0: Initial checks and setup for arguments
        _, score_c = get_scoring_method(scoring_method, self.data, self.use_cache)
        score_fn = score_c.local_score

        # Step 1: Initialize an empty model.
        current_model = PDAG()
        current_model.add_nodes_from(list(self.data.columns))
        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(self.data.columns)

        expert_knowledge._orient_temporal_forbidden_edges(
            current_model, only_edges=False
        )
        all_nodes = list(self.data.columns)

        # Step 2: Forward step: Iteratively add edges till score stops improving.
        while True:
            potential_edges = self._legal_edge_additions(
                current_model, expert_knowledge
            )
            print("At star of step\n", current_model.edges, "\n")
            score_deltas = np.zeros(len(potential_edges))
            insertion_ops = []
            for index, (u, v) in enumerate(potential_edges):
                # current_parents = current_model.get_parents(v)
                insertion_op = self._score_valid_insertions(
                    u, v, current_model, score_fn
                )
                if insertion_op == []:
                    score_deltas[index] = 0
                    insertion_ops.append(None)
                else:
                    score_deltas[index] = max(insertion_op)[0]
                    insertion_ops.append(max(insertion_op))

            print("\n", potential_edges, score_deltas, "\n")

            if (len(potential_edges) == 0) or (np.all(score_deltas < min_improvement)):
                break

            edge_to_add = potential_edges[np.argmax(score_deltas)]
            op_to_add = insertion_ops[np.argmax(score_deltas)]

            # print("2.\n", current_model.edges)
            print("befor", current_model.edges, "\n")
            current_model = self.insert(
                edge_to_add[0], edge_to_add[1], op_to_add[4], current_model
            )
            print("after", current_model.edges, "\n")

            # print('\n1 step', current_model.edges)

            if not self._is_dag(current_model):
                new_model = current_model.to_dag()
                new_model.add_nodes_from(current_model.nodes)
                # print('\n1 step', new_model.edges, '\n')
                current_model = new_model.to_pdag()
            else:
                current_model = DAG(ebunch=current_model.edges).to_pdag()
            current_model.add_nodes_from(all_nodes)

            print("Pdag after 1 step", current_model.edges, "\n")
            if debug:
                logger.info(
                    f"Adding edge {edge_to_add[0]} -> {edge_to_add[1]}. Improves score by: {score_deltas.max()}"
                )

        # Step 3: Backward Step: Iteratively remove edges till score stops improving.
        while False:
            # print(current_model.edges)

            potential_removals = self._legal_edge_removals(
                current_model, expert_knowledge
            )

            # print("\nremovals\n", potential_removals)
            score_deltas = np.zeros(len(potential_removals))
            deletion_ops = []

            for index, (u, v) in enumerate(potential_removals):
                # print("\nloop-edges\n", current_model.edges)
                deletion_op = self._score_valid_deletions(u, v, current_model, score_fn)
                if deletion_op == []:
                    score_deltas[index] = 0
                    deletion_ops.append(None)
                else:
                    score_deltas[index] = max(deletion_op)[0]
                    deletion_ops.append(max(deletion_op))

            # print("Reached!!!")

            print(score_deltas)
            if (len(potential_removals) == 0) or (
                np.all(score_deltas < min_improvement)
            ):
                break

            edge_to_remove = potential_removals[np.argmax(score_deltas)]
            op_to_delete = deletion_ops[np.argmax(score_deltas)]
            current_model = self.delete(
                edge_to_remove[0], edge_to_remove[1], op_to_delete[4], current_model
            )

            if not self._is_dag(current_model):
                new_model = current_model.to_dag()
                new_model.add_nodes_from(current_model.nodes)
                # print('\n1 step', new_model.edges, '\n')
                current_model = new_model.to_pdag()
            else:
                current_model = DAG(ebunch=current_model.edges).to_pdag()
            current_model.add_nodes_from(all_nodes)

            print("Pdag after 1 step", current_model.edges)
            if debug:
                logger.info(
                    f"Removing edge {edge_to_remove[0]} -> {edge_to_remove[1]}. Improves score by: {score_deltas.max()}"
                )

        # Step 4: Flip Edges: Iteratively try to flip edges till score stops improving.
        while False:
            potential_turns = self._legal_edge_turns(current_model, expert_knowledge)
            score_deltas = np.zeros(len(potential_turns))
            turn_ops = []

            for index, (u, v) in enumerate(potential_turns):
                turn_op = self._score_valid_turns(u, v, current_model, score_fn)
                if turn_op == []:
                    score_deltas[index] = 0
                    turn_ops.append(None)
                else:
                    score_deltas[index] = max(turn_op)[0]
                    turn_ops.append(max(turn_op))

            if (len(potential_turns) == 0) or (np.all(score_deltas < min_improvement)):
                break

            edge_to_flip = potential_turns[np.argmax(score_deltas)]
            op_to_turn = turn_ops[np.argmax(score_deltas)]
            current_model = self.turn(
                edge_to_flip[0], edge_to_flip[1], op_to_turn[4], current_model
            )

            # print(current_model.edges)
            if not self._is_dag(current_model):
                new_model = current_model.to_dag()
                new_model.add_nodes_from(current_model.nodes)
                print("\n3 step", new_model.edges, "\n")
                current_model = new_model.to_pdag()
            else:
                current_model = DAG(ebunch=current_model.edges).to_pdag()

            if debug:
                logger.info(
                    f"Fliping edge {edge_to_flip[1]} -> {edge_to_flip[0]}. Improves score by: {score_deltas.max()}"
                )

        # Step 5: Return the model.
        return current_model
