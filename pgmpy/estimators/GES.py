from itertools import combinations

import networkx as nx
import numpy as np

from pgmpy import config
from pgmpy.base import DAG
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

    GES is a score-based causal discovery / structure learning algorithm that operates on
    equivalence classes of DAGs. It typically involves two phases:
        1. Forward Equivalence Search (FES): Greedily applies Chickering's "Insert" operator
           to add edges and move between equivalence classes, improving the model score.
        2. Backward Equivalence Search (BES): Greedily applies Chickering's "Delete" operator
           to remove edges and move between equivalence classes, improving the model score.

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
    Meek, Christopher. "Causal inference and causal explanation with background knowledge." Proceedings of the Eleventh conference on Uncertainty in artificial intelligence. Morgan Kaufmann Publishers Inc., 1995.
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache
        super(GES, self).__init__(data=data, **kwargs)

    def _calculate_score_delta_insert(
        self, current_model, g_prime, score_fn, X_node, Y_node, T
    ):
        """
        Calculates the score change for an Insert(X, Y, T) operation.
        Delta = Score(G') - Score(G)
              = sum_{V} (local_score_G'(V, Pa_G'(V)) - local_score_G(V, Pa_G(V)))
        We only need to sum over nodes whose parent sets change: Y, X, and Z in T (if X-Z edge was removed).
        """
        score_delta = 0

        # Node Y
        pa_Y_old = frozenset(current_model.get_parents(Y_node))
        pa_Y_new = frozenset(g_prime.get_parents(Y_node))
        if pa_Y_old != pa_Y_new:
            score_delta += score_fn(Y_node, list(pa_Y_new)) - score_fn(
                Y_node, list(pa_Y_old)
            )

        # Node X (if any Z in T was a parent of X, and Z->X was removed)
        pa_X_old = frozenset(current_model.get_parents(X_node))
        pa_X_new = frozenset(g_prime.get_parents(X_node))
        if pa_X_old != pa_X_new:
            score_delta += score_fn(X_node, list(pa_X_new)) - score_fn(
                X_node, list(pa_X_old)
            )

        # Nodes Z in T (if X was a parent of Z, and X->Z was removed)
        for Z_node_in_T in T:
            if current_model.has_edge(X_node, Z_node_in_T):  # X->Z_node_in_T was in G
                pa_Z_old = frozenset(current_model.get_parents(Z_node_in_T))
                pa_Z_new = frozenset(g_prime.get_parents(Z_node_in_T))
                if pa_Z_old != pa_Z_new:
                    score_delta += score_fn(Z_node_in_T, list(pa_Z_new)) - score_fn(
                        Z_node_in_T, list(pa_Z_old)
                    )
        return score_delta

    def _get_best_insert_operation(self, current_model, score_fn, expert_knowledge):
        nodes = list(current_model.nodes())
        best_op_details = None
        max_score_delta = -np.inf

        candidate_ops = []

        for X_node in nodes:
            for Y_node in nodes:
                if (
                    X_node == Y_node
                    or current_model.has_edge(X_node, Y_node)
                    or current_model.has_edge(Y_node, X_node)
                ):
                    continue

                if (X_node, Y_node) in expert_knowledge.forbidden_edges:
                    continue

                # T is a subset of Neighbors_G(X) \ {Y}
                neighbors_X = set(current_model.predecessors(X_node)) | set(
                    current_model.successors(X_node)
                )
                neighbors_X_without_Y = list(neighbors_X - {Y_node})

                for k in range(len(neighbors_X_without_Y) + 1):
                    for T_set_nodes in combinations(neighbors_X_without_Y, k):
                        T = list(T_set_nodes)

                        forbidden_in_T = any(
                            (Z_node, Y_node) in expert_knowledge.forbidden_edges
                            for Z_node in T
                        )
                        if forbidden_in_T:
                            continue

                        g_prime = current_model.copy()
                        g_prime.add_edge(X_node, Y_node)
                        edges_to_remove_from_X_T = []
                        for Z_in_T in T:
                            g_prime.add_edge(Z_in_T, Y_node)
                            if g_prime.has_edge(X_node, Z_in_T):
                                edges_to_remove_from_X_T.append((X_node, Z_in_T))
                            if g_prime.has_edge(Z_in_T, X_node):
                                edges_to_remove_from_X_T.append((Z_in_T, X_node))

                        for u, v_node in edges_to_remove_from_X_T:
                            if g_prime.has_edge(u, v_node):
                                g_prime.remove_edge(u, v_node)

                        if not nx.is_directed_acyclic_graph(g_prime):
                            continue

                        valid_cond2 = all(
                            Z_neighbor in T
                            for Z_neighbor in neighbors_X
                            if Z_neighbor != Y_node
                            and g_prime.has_edge(Z_neighbor, Y_node)
                        )
                        if not valid_cond2:
                            continue

                        current_score_delta = self._calculate_score_delta_insert(
                            current_model, g_prime, score_fn, X_node, Y_node, T
                        )
                        candidate_ops.append(
                            (current_score_delta, X_node, Y_node, tuple(T))
                        )

        if not candidate_ops:
            return None, 0

        candidate_ops.sort(key=lambda x: x[0], reverse=True)
        best_delta, best_X, best_Y, best_T_tuple = candidate_ops[0]

        # Return if positive delta, actual check against min_improvement done in main loop
        if best_delta > 0:
            return (best_X, best_Y, list(best_T_tuple)), best_delta
        return None, 0

    def _apply_insert(self, current_model, X, Y, T):
        current_model.add_edge(X, Y)
        edges_to_remove = []
        for Z_node in T:
            current_model.add_edge(Z_node, Y)
            if current_model.has_edge(X, Z_node):
                edges_to_remove.append((X, Z_node))
            if current_model.has_edge(Z_node, X):
                edges_to_remove.append((Z_node, X))

        for u, v in edges_to_remove:
            if current_model.has_edge(u, v):
                current_model.remove_edge(u, v)

    def _calculate_score_delta_delete(
        self, current_model, g_prime, score_fn, X_node, Y_node, H
    ):
        """
        Calculates the score change for a Delete(X, Y, H) operation.
        Affected nodes: Y, and Z in H.
        """
        score_delta = 0

        # Node Y
        pa_Y_old = frozenset(current_model.get_parents(Y_node))
        pa_Y_new = frozenset(g_prime.get_parents(Y_node))
        if pa_Y_old != pa_Y_new:
            score_delta += score_fn(Y_node, list(pa_Y_new)) - score_fn(
                Y_node, list(pa_Y_old)
            )

        # Nodes Z in H (Pa(Z) changes as X becomes a parent, Z->Y is removed)
        for Z_node_in_H in H:
            pa_Z_old = frozenset(current_model.get_parents(Z_node_in_H))
            pa_Z_new = frozenset(g_prime.get_parents(Z_node_in_H))
            if pa_Z_old != pa_Z_new:
                score_delta += score_fn(Z_node_in_H, list(pa_Z_new)) - score_fn(
                    Z_node_in_H, list(pa_Z_old)
                )
        return score_delta

    def _get_best_delete_operation(self, current_model, score_fn, expert_knowledge):
        best_op_details = None
        max_score_delta = -np.inf
        candidate_ops = []

        for X_node, Y_node in list(current_model.edges()):
            if (X_node, Y_node) in expert_knowledge.required_edges:
                continue

            # H is a subset of Pa_G(Y) \ {X}
            parents_Y_without_X = list(
                set(current_model.get_parents(Y_node)) - {X_node}
            )

            for k in range(len(parents_Y_without_X) + 1):
                for H_set_nodes in combinations(parents_Y_without_X, k):
                    H = list(H_set_nodes)

                    forbidden_in_H = any(
                        (X_node, Z_node) in expert_knowledge.forbidden_edges
                        for Z_node in H
                    )
                    if forbidden_in_H:
                        continue

                    g_prime = current_model.copy()
                    if g_prime.has_edge(
                        X_node, Y_node
                    ):  # Ensure edge exists before removing
                        g_prime.remove_edge(X_node, Y_node)
                    else:  # Should not happen if iterating current_model.edges()
                        continue

                    edges_to_remove_from_Z_Y = []
                    for Z_in_H in H:
                        g_prime.add_edge(X_node, Z_in_H)  # X becomes parent of Z
                        if g_prime.has_edge(Z_in_H, Y_node):  # Z was parent of Y
                            edges_to_remove_from_Z_Y.append((Z_in_H, Y_node))

                    for u, v_node in edges_to_remove_from_Z_Y:
                        if g_prime.has_edge(u, v_node):
                            g_prime.remove_edge(u, v_node)

                    if not nx.is_directed_acyclic_graph(g_prime):
                        continue

                    valid_cond2 = all(
                        Z_parent_of_Y in H
                        for Z_parent_of_Y in parents_Y_without_X
                        if g_prime.has_edge(X_node, Z_parent_of_Y)
                    )
                    if not valid_cond2:
                        continue

                    current_score_delta = self._calculate_score_delta_delete(
                        current_model, g_prime, score_fn, X_node, Y_node, H
                    )
                    candidate_ops.append(
                        (current_score_delta, X_node, Y_node, tuple(H))
                    )

        if not candidate_ops:
            return None, 0

        candidate_ops.sort(key=lambda x: x[0], reverse=True)
        best_delta, best_X, best_Y, best_H_tuple = candidate_ops[0]

        if best_delta > 0:
            return (best_X, best_Y, list(best_H_tuple)), best_delta
        return None, 0

    def _apply_delete(self, current_model, X, Y, H):
        if current_model.has_edge(X, Y):  # Ensure edge exists
            current_model.remove_edge(X, Y)

        edges_to_remove = []  # Store Z->Y edges that need to be removed
        for Z_node in H:
            current_model.add_edge(X, Z_node)  # X becomes parent of Z
            if current_model.has_edge(Z_node, Y):  # If Z was parent of Y
                edges_to_remove.append((Z_node, Y))

        for u, v in edges_to_remove:
            if current_model.has_edge(u, v):  # Ensure edge Z->Y exists before removing
                current_model.remove_edge(u, v)

    def estimate(
        self,
        scoring_method="bic-d",
        expert_knowledge=None,
        min_improvement=1e-6,
        debug=False,
    ):
        """
        Estimates the DAG (representing an equivalence class) from the data using GES.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation. Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
            be an instance of `StructureScore`.

        expert_knowledge: pgmpy.estimators.ExpertKnowledge instance (default: None)
            Expert knowledge to be used with the algorithm. Expert knowledge
            allows specification of required and forbidden edges, as well as temporal
            order of nodes.

        min_improvement: float
            The operation (Insert or Delete) would only be performed if the
            model score improves by at least `min_improvement`.

        debug: bool
            If True, logs the operations performed.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` representing an equivalence class found by GES.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.estimators import GES
        >>> # Simulate some sample data
        >>> model = get_example_model('alarm')
        >>> data = model.simulate(int(1e3))
        >>> est = GES(data)
        >>> # Learn model structure (GES may not recover the exact alarm structure, but a CPDAG in the same equivalence class or a nearby one)
        >>> learned_model = est.estimate(scoring_method='bic-d')
        >>> print(len(learned_model.nodes()), len(learned_model.edges()))
        """

        # Step 0: Initial checks and setup
        _, score_c = get_scoring_method(scoring_method, self.data, self.use_cache)
        score_fn = score_c.local_score

        # Step 1: Initialize an empty model.
        current_model = DAG()
        current_model.add_nodes_from(list(self.data.columns))
        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(self.data.columns)

        expert_knowledge._orient_temporal_forbidden_edges(
            current_model, only_edges=False
        )

        # Step 2: Forward Equivalence Search (FES)
        if debug:
            logger.info("Starting Forward Equivalence Search (FES) phase.")
        while True:
            op_details, score_delta = self._get_best_insert_operation(
                current_model, score_fn, expert_knowledge
            )

            if op_details and score_delta > min_improvement:
                X, Y, T = op_details
                self._apply_insert(current_model, X, Y, T)
                if debug:
                    logger.info(
                        f"FES: Applied Insert({X}, {Y}, {T}). Score improved by: {score_delta:.4f}"
                    )
            else:
                if debug and op_details:
                    logger.info(
                        f"FES: Best Insert op score delta {score_delta:.4f} <= min_improvement {min_improvement}. Stopping FES."
                    )
                elif debug:
                    logger.info(
                        "FES: No improving Insert operation found. Stopping FES."
                    )
                break

        if debug:
            logger.info("Finished Forward Equivalence Search (FES) phase.")

        # Step 3: Backward Equivalence Search (BES)
        if debug:
            logger.info("Starting Backward Equivalence Search (BES) phase.")
        while True:
            op_details, score_delta = self._get_best_delete_operation(
                current_model, score_fn, expert_knowledge
            )

            if op_details and score_delta > min_improvement:
                X, Y, H = op_details
                self._apply_delete(current_model, X, Y, H)
                if debug:
                    logger.info(
                        f"BES: Applied Delete({X}, {Y}, {H}). Score improved by: {score_delta:.4f}"
                    )
            else:
                if debug and op_details:
                    logger.info(
                        f"BES: Best Delete op score delta {score_delta:.4f} <= min_improvement {min_improvement}. Stopping BES."
                    )
                elif debug:
                    logger.info(
                        "BES: No improving Delete operation found. Stopping BES."
                    )
                break
        if debug:
            logger.info("Finished Backward Equivalence Search (BES) phase.")

        # Step 4: Return the model.
        return current_model
