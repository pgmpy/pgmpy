import math
from collections import deque

import numpy as np

from pgmpy.base import DAG
from pgmpy.metrics import BaseSupervisedMetric


def _compute_path_matrix(graph: np.ndarray) -> np.ndarray:
    """Compute the directed transitive closure, including the diagonal."""
    n_nodes = graph.shape[0]
    # An empty graph has no closure to compute, and ``math.log2(0)`` would raise below.
    if n_nodes == 0:
        return np.empty((0, 0), dtype=bool)

    # Start from (I + G): every node reaches its direct children and, via the identity, itself.
    # The reflexive diagonal matters later, where ``path_matrix[c, target]`` must hold for c == target.
    path_matrix = graph.astype(bool) | np.eye(n_nodes, dtype=bool)

    # Each squaring doubles the path length covered, so ceil(log2(p)) rounds reach every path: the
    # longest simple path in a p-node DAG has p - 1 edges. On a bool array ``@`` is logical matrix
    # multiplication (+ is OR, * is AND), so "a reaches b and b reaches c" composes into "a reaches c".
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
    """Implement Algorithm 2 (``rondp``) from Peters and Buehlmann (2015).

    Answers part (b) of the adjustment criterion: which nodes are reachable from ``source`` along a
    path that is open given ``conditioned`` *and* is not a directed causal path. The "not causal"
    restriction is why this cannot delegate to ``DAG.active_trail_nodes``, which models plain
    d-separation and happily returns nodes reached along directed paths.
    """
    n_nodes = graph.shape[0]

    # Nodes with a directed path into the conditioning set. Because ``path_matrix`` is reflexive the
    # conditioned nodes are their own ancestors, so this is exactly the set of colliders that
    # conditioning opens: a collider is active iff it or one of its descendants is conditioned on.
    ancestors_of_conditioned = (
        path_matrix[:, conditioned].any(axis=1) if conditioned.any() else np.zeros(n_nodes, dtype=bool)
    )

    # State doubling. Each node gets two indices: v means "arrived via an edge pointing into v"
    # and n_nodes + v means "arrived via an edge pointing out of v". Whether a path may continue
    # through a node depends on that arrival direction (the collider rule), so the direction has to
    # be part of the state. ``reachability_matrix`` is the edge relation over those 2p states and is
    # only transitively closed once the traversal below has finished building it.
    reachability_matrix = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=bool)
    reachable_on_non_directed_path = np.zeros(2 * n_nodes, dtype=bool)
    reachable_later = []

    parents_of_source = graph[:, source].astype(bool)
    children_of_source = graph[source, :].astype(bool)
    nodes_to_check = deque(np.flatnonzero(children_of_source).tolist() + np.flatnonzero(parents_of_source).tolist())

    # Seed only the parents of the source, in their "arrived via an outgoing edge" state: stepping
    # source <- parent is already non-directed. The children are deliberately left unseeded because
    # source -> child is a causal step, and that asymmetry is the whole "non-directed" restriction.
    reachable_on_non_directed_path[n_nodes:] = parents_of_source

    # Work on a copy with the edges incident to the source removed, so no path can loop back through
    # it -- the paper's requirement that paths be "proper".
    graph = graph.astype(bool, copy=True)
    graph[parents_of_source, source] = False
    graph[source, children_of_source] = False
    checked = np.zeros(n_nodes, dtype=bool)

    while nodes_to_check:
        current_node = nodes_to_check.popleft()
        if checked[current_node]:
            continue
        checked[current_node] = True

        # The arrival state and the departure direction together fix which d-separation role
        # ``current_node`` (written c below) plays, so each of the four combinations gets exactly one
        # transition. Writing a for a parent and b, d for children:
        #
        #   from     to              walk           role at c    guarded by
        #   c        child           a -> c -> d    chain        not conditioned[c]
        #   n+c      child           b <- c -> d    FORK         not conditioned[c]
        #   n+c      n+parent        b <- c <- a    chain        not conditioned[c]
        #   c        n+parent        a -> c <- a'   collider     ancestors_of_conditioned[c]
        #
        # Chains and forks block under the same condition, which is why all three non-collider
        # transitions share the single ``not conditioned`` guard rather than being cased apart.
        #
        # Every one of these transitions is recorded twice, once from each end of the edge: the
        # parents block below writes them while visiting the child, the children block while
        # visiting the parent. The duplication is harmless but total -- deleting either block
        # wholesale leaves the results unchanged, because parents are always queued (a conditioned
        # node is its own ancestor, so one of the two guards below always fires) and therefore a
        # visited node's parents are visited too. Kept as-is to match Algorithm 2 as published.

        # --- Parents of the current node ---
        parents = graph[:, current_node]
        parent_indices = np.flatnonzero(parents)
        unconditioned_parents = np.flatnonzero(parents & ~conditioned)

        # Mirror of the two rows above, recorded from this end of the edge: these are the chain and
        # fork transitions at the *parent*, which is why both arrival states are set and the guard is
        # on the parent being unconditioned rather than on ``current_node``.
        reachability_matrix[unconditioned_parents, current_node] = True
        reachability_matrix[n_nodes + unconditioned_parents, current_node] = True

        # Collider at ``current_node``: traversal upwards into the parents is only unblocked when the
        # node is an ancestor of the conditioning set. Both the deferred bookkeeping and the queue
        # push below belong inside this guard -- doing them unconditionally opens blocked paths.
        if ancestors_of_conditioned[current_node]:
            reachability_matrix[current_node, n_nodes + parent_indices] = True
            # Reached from the source along a directed path that avoids conditioned tails, so the
            # turn-around at this collider is settled in the correction pass rather than now.
            if path_matrix_without_conditioned_tails[source, current_node]:
                reachable_later.extend((current_node, parent) for parent in parent_indices)
            nodes_to_check.extend(parent for parent in parent_indices if not checked[parent])

        # Chain at ``current_node``, continuing upwards: b <- c <- a. The fork sharing this guard is
        # in the children block below, since it departs towards a child instead.
        if not conditioned[current_node]:
            reachability_matrix[n_nodes + current_node, n_nodes + parent_indices] = True
            nodes_to_check.extend(parent for parent in parent_indices if not checked[parent])

        # --- Children of the current node ---
        children = graph[current_node, :]
        child_indices = np.flatnonzero(children)
        unconditioned_children = np.flatnonzero(children & ~conditioned)
        # Mirror of the chain-upwards row, recorded at the child end.
        reachability_matrix[n_nodes + unconditioned_children, n_nodes + current_node] = True

        # A child that is an ancestor of the conditioning set is an activated collider, so arriving
        # at the current node from it keeps the path open. Mirror of the collider row.
        relevant_children = children & ancestors_of_conditioned
        relevant_child_indices = np.flatnonzero(relevant_children)
        reachability_matrix[relevant_child_indices, n_nodes + current_node] = True
        reachable_later.extend(
            (child, current_node)
            for child in np.flatnonzero(relevant_children & path_matrix_without_conditioned_tails[source, :])
        )

        if not conditioned[current_node]:
            # Chain continuing downwards: a -> c -> d.
            reachability_matrix[current_node, child_indices] = True
            # Fork: b <- c -> d. This is the non-collider case that departs towards a child having
            # arrived from another child, and it blocks on exactly the same condition as the chains.
            reachability_matrix[n_nodes + current_node, child_indices] = True
            nodes_to_check.extend(child for child in child_indices if not checked[child])

    # Close the relation over the 2p states, then propagate outwards from the seeded states.
    reachability_matrix = _compute_path_matrix(reachability_matrix)
    reachable_on_non_directed_path |= reachability_matrix[reachable_on_non_directed_path, :].any(axis=0)

    # Correction pass. These nodes were reached from the source along a directed path that then turns
    # around at an activated collider -- a legitimately non-causal continuation. Mark them reachable,
    # but first delete the reachability edges that would otherwise let the purely directed prefix
    # count as a non-directed path in its own right, then propagate again.
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

    # Collapse the doubled state space: a node counts as reached if either arrival state was reached.
    return reachable_on_non_directed_path[:n_nodes] | reachable_on_non_directed_path[n_nodes:]


def _sid_matrix(true_graph: np.ndarray, est_graph: np.ndarray) -> np.ndarray:
    """Return the ordered pairs whose intervention distributions are incorrect.

    Implements Algorithm 1. For each ordered pair (source, target) the question is whether the
    estimated graph's parent set of ``source`` is a valid adjustment set for that pair in the true
    graph, which the generalized adjustment criterion answers with two conditions:

    (a) no element of the adjustment set descends from a node (other than ``source``) that lies on a
        directed path from ``source`` to ``target``, and
    (b) the adjustment set blocks every non-causal path between them.
    """
    n_nodes = true_graph.shape[0]
    path_matrix = _compute_path_matrix(true_graph)
    incorrect_interventions = np.zeros((n_nodes, n_nodes), dtype=bool)

    for source in range(n_nodes):
        true_parents = true_graph[:, source].astype(bool)
        est_parents = est_graph[:, source].astype(bool)
        # Matching parent sets make the adjustment set the true back-door set, which is valid for
        # every target, so the whole source can be skipped.
        if np.array_equal(true_parents, est_parents):
            continue

        # Drop the outgoing edges of the adjustment set; the correction pass in Algorithm 2 uses this
        # to tell a directed path that avoids conditioned tails from one that does not. With an empty
        # adjustment set nothing changes, so reuse the closure already computed above.
        if est_parents.any():
            graph_without_conditioned_tails = true_graph.copy()
            graph_without_conditioned_tails[est_parents, :] = 0
            path_matrix_without_conditioned_tails = _compute_path_matrix(graph_without_conditioned_tails)
        else:
            path_matrix_without_conditioned_tails = path_matrix

        # One traversal per source answers condition (b) for every target at once.
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

            # The target sits in the adjustment set, so the estimated intervention distribution
            # collapses to the marginal p(x_target). That is right exactly when the source really has
            # no causal effect on it.
            if est_effect_is_null:
                incorrect_interventions[source, target] = not true_effect_is_null
                continue

            # Condition (a), relevant only when a causal path exists: reject the adjustment set if it
            # contains a descendant of a child of the source that still reaches the target. The
            # reflexive diagonal of ``path_matrix`` makes a child that *is* the target count too.
            # This is the "never adjust for a mediator or its descendants" rule.
            if path_matrix[source, target]:
                children_on_causal_path = true_graph[source, :].astype(bool) & path_matrix[:, target]
                if path_matrix[children_on_causal_path][:, est_parents].any():
                    incorrect_interventions[source, target] = True
                    continue

            # Condition (b): an open non-causal path is left unblocked by the adjustment set.

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
        # Take the node order once and reuse it as the ``nodelist`` for both conversions, so the two
        # matrices stay row- and column-aligned even when the graphs enumerate their nodes differently.
        nodes = list(true_causal_graph.nodes())
        true_adjacency = true_causal_graph.to_adjacency(encoding="binary", nodelist=nodes).to_numpy(dtype=bool)
        est_adjacency = est_causal_graph.to_adjacency(encoding="binary", nodelist=nodes).to_numpy(dtype=bool)
        # Each True cell is one ordered pair with a falsely inferred intervention distribution.
        return int(_sid_matrix(true_adjacency, est_adjacency).sum())
