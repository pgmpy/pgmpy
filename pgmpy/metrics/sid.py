import math
from collections import deque

import numpy as np

from pgmpy.base import DAG
from pgmpy.metrics import BaseSupervisedMetric


def _compute_path_matrix(graph: np.ndarray) -> np.ndarray:
    """Compute the directed transitive closure, including the diagonal."""
    n_nodes = graph.shape[0]
    if n_nodes == 0:
        return np.empty((0, 0), dtype=bool)

    path_matrix = graph.astype(bool) | np.eye(n_nodes, dtype=bool)
    for _ in range(math.ceil(math.log2(n_nodes))):
        path_matrix = path_matrix @ path_matrix
    return path_matrix


def _reachable_on_non_directed_path(
    graph: np.ndarray,
    source: int,
    conditioned: np.ndarray,
    path_matrix: np.ndarray,
    path_matrix_without_conditioned_tails: np.ndarray,
) -> np.ndarray:
    """Implement Algorithm 2 (``rondp``) from Peters and Buehlmann (2015)."""
    n_nodes = graph.shape[0]
    ancestors_of_conditioned = (
        path_matrix[:, conditioned].any(axis=1) if conditioned.any() else np.zeros(n_nodes, dtype=bool)
    )
    reachability_matrix = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=bool)
    reachable_on_non_directed_path = np.zeros(2 * n_nodes, dtype=bool)
    reachable_later = []

    parents_of_source = graph[:, source].astype(bool)
    children_of_source = graph[source, :].astype(bool)
    nodes_to_check = deque(np.flatnonzero(children_of_source).tolist() + np.flatnonzero(parents_of_source).tolist())
    reachable_on_non_directed_path[n_nodes:] = parents_of_source

    graph = graph.astype(bool, copy=True)
    graph[parents_of_source, source] = False
    graph[source, children_of_source] = False
    checked = np.zeros(n_nodes, dtype=bool)

    while nodes_to_check:
        current_node = nodes_to_check.popleft()
        if checked[current_node]:
            continue
        checked[current_node] = True

        parents = graph[:, current_node]
        parent_indices = np.flatnonzero(parents)
        unconditioned_parents = np.flatnonzero(parents & ~conditioned)
        reachability_matrix[unconditioned_parents, current_node] = True
        reachability_matrix[n_nodes + unconditioned_parents, current_node] = True

        if ancestors_of_conditioned[current_node]:
            reachability_matrix[current_node, n_nodes + parent_indices] = True
            if path_matrix_without_conditioned_tails[source, current_node]:
                reachable_later.extend((current_node, parent) for parent in parent_indices)
            nodes_to_check.extend(parent for parent in parent_indices if not checked[parent])

        if not conditioned[current_node]:
            reachability_matrix[n_nodes + current_node, n_nodes + parent_indices] = True
            nodes_to_check.extend(parent for parent in parent_indices if not checked[parent])

        children = graph[current_node, :]
        child_indices = np.flatnonzero(children)
        unconditioned_children = np.flatnonzero(children & ~conditioned)
        reachability_matrix[n_nodes + unconditioned_children, n_nodes + current_node] = True

        relevant_children = children & ancestors_of_conditioned
        relevant_child_indices = np.flatnonzero(relevant_children)
        reachability_matrix[relevant_child_indices, n_nodes + current_node] = True
        reachable_later.extend(
            (child, current_node)
            for child in np.flatnonzero(relevant_children & path_matrix_without_conditioned_tails[source, :])
        )

        if not conditioned[current_node]:
            reachability_matrix[current_node, child_indices] = True
            reachability_matrix[n_nodes + current_node, child_indices] = True
            nodes_to_check.extend(child for child in child_indices if not checked[child])

    reachability_matrix = _compute_path_matrix(reachability_matrix)
    reachable_on_non_directed_path |= reachability_matrix[reachable_on_non_directed_path, :].any(axis=0)

    if reachable_later:
        for reachable_through, new_reachable in reachable_later:
            reachable_on_non_directed_path[n_nodes + new_reachable] = True
            reachability_matrix[
                [new_reachable, new_reachable, n_nodes + new_reachable, n_nodes + new_reachable],
                [
                    reachable_through,
                    n_nodes + reachable_through,
                    reachable_through,
                    n_nodes + reachable_through,
                ],
            ] = False

        reachable_on_non_directed_path |= reachability_matrix[reachable_on_non_directed_path, :].any(axis=0)

    return reachable_on_non_directed_path[:n_nodes] | reachable_on_non_directed_path[n_nodes:]


def _sid_matrix(true_graph: np.ndarray, est_graph: np.ndarray) -> np.ndarray:
    """Return the ordered pairs whose intervention distributions are incorrect."""
    n_nodes = true_graph.shape[0]
    path_matrix = _compute_path_matrix(true_graph)
    incorrect_interventions = np.zeros((n_nodes, n_nodes), dtype=bool)

    for source in range(n_nodes):
        true_parents = true_graph[:, source].astype(bool)
        est_parents = est_graph[:, source].astype(bool)
        if np.array_equal(true_parents, est_parents):
            continue

        if est_parents.any():
            graph_without_conditioned_tails = true_graph.copy()
            graph_without_conditioned_tails[est_parents, :] = 0
            path_matrix_without_conditioned_tails = _compute_path_matrix(graph_without_conditioned_tails)
        else:
            path_matrix_without_conditioned_tails = path_matrix

        reachable_on_non_directed_path = _reachable_on_non_directed_path(
            true_graph,
            source,
            est_parents,
            path_matrix,
            path_matrix_without_conditioned_tails,
        )

        for target in range(n_nodes):
            if source == target:
                continue

            true_effect_is_null = not path_matrix[source, target]
            est_effect_is_null = est_parents[target]
            if est_effect_is_null:
                incorrect_interventions[source, target] = not true_effect_is_null
                continue

            if path_matrix[source, target]:
                children_on_causal_path = true_graph[source, :].astype(bool) & path_matrix[:, target]
                if path_matrix[children_on_causal_path][:, est_parents].any():
                    incorrect_interventions[source, target] = True
                    continue

            incorrect_interventions[source, target] = reachable_on_non_directed_path[target]

    return incorrect_interventions


class SID(BaseSupervisedMetric):
    r"""
    Computes the Structural Intervention Distance (SID) between two DAGs.

    SID counts the ordered pairs of variables for which the estimated graph
    falsely infers an intervention distribution with respect to the true graph.
    Unlike SHD, SID is asymmetric and can be zero when the graphs differ.

    Notes
    -----
    The implementation follows Algorithms 1 and 2 in Appendix F of the paper.
    Graphs are converted to aligned adjacency matrices once, and the path and
    state-doubled reachability computations use NumPy arrays.

    As in the reference implementation, the transitive closure is recomputed for
    each source node, so the cost grows roughly with the fourth power of the
    number of nodes: a 100-node pair takes a few seconds and a 200-node pair
    around a minute.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from pgmpy.metrics import SID
    >>> true_dag = DAG([(1, 2)])
    >>> est_dag = DAG([(2, 1)])
    >>> SID()(true_causal_graph=true_dag, est_causal_graph=est_dag)
    2

    References
    ----------
    - :footcite:t:`peters_buhlmann_2015`

    """

    _tags = {
        "name": "SID",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "is_default": False,
    }

    def _evaluate(self, true_causal_graph: DAG, est_causal_graph: DAG) -> int:
        # `BaseSupervisedMetric.evaluate` has already checked the types and the shared node set.
        nodes = list(true_causal_graph.nodes())
        true_adjacency = true_causal_graph.to_adjacency(encoding="binary", nodelist=nodes).to_numpy(dtype=bool)
        est_adjacency = est_causal_graph.to_adjacency(encoding="binary", nodelist=nodes).to_numpy(dtype=bool)
        return int(_sid_matrix(true_adjacency, est_adjacency).sum())
