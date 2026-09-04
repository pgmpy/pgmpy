from itertools import combinations

import numpy as np
import pandas as pd

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import BaseSupervisedMetric


def _list_of_index(bool_np_array):
    return np.where(bool_np_array)[0].tolist()


def _compute_path_matrix(G):
    """
    Computes the transitive closure (path matrix) of a DAG adjacency matrix G.
    path_matrix[i, j] is True if there is a directed path from i to j (including i == j).
    """
    p = len(G)
    path_matrix = np.array(G + np.eye(p), dtype=bool)

    k = int(np.ceil(np.log(p) / np.log(2))) if p > 1 else 1
    for _ in range(k):
        path_matrix = np.matmul(path_matrix, path_matrix)

    return path_matrix


def _compute_or_reuse_path_matrix(G, path_matrix, cond_set):
    """
    Computes path matrix of G with nodes in cond_set cut (no outgoing edges),
    or reuses path_matrix if cond_set is empty.
    """
    if not np.any(cond_set):
        return path_matrix

    G_tilde = np.array(G, copy=True)
    G_tilde[cond_set, :] = 0

    return _compute_path_matrix(G_tilde)


def _reachable_on_non_directed_path(G_orig, i, cond_set, path_matrix, path_matrix_2):
    """
    Algorithm 2 from Peters & Bühlmann (2015): Reachability on non-directed paths.
    Computes which nodes are reachable from node i via non-directed paths in G given cond_set.
    """
    p = len(G_orig)
    is_ancestor_of_cond_set = np.any(path_matrix[:, cond_set], axis=1) > 0

    reachability_matrix = np.zeros((2 * p, 2 * p), dtype=bool)
    reachable_on_non_causal_path_later = np.zeros((2, 2), dtype=np.int64)
    is_parent_i = G_orig[:, i] > 0
    is_child_i = G_orig[i, :] > 0

    queue_of_nodes_to_check = _list_of_index(is_parent_i) + _list_of_index(is_child_i)
    reachable_nodes = np.concatenate((is_child_i, is_parent_i))
    reachable_on_non_causal_path = np.concatenate(([False] * p, is_parent_i))

    G = np.array(G_orig, copy=True)
    G[is_parent_i, i] = 0
    G[i, is_child_i] = 0

    node_is_already_checked = np.zeros(p, dtype=bool)

    while queue_of_nodes_to_check:
        current_node = queue_of_nodes_to_check.pop()
        if node_is_already_checked[current_node]:
            continue

        node_is_already_checked[current_node] = True
        Pa = G[:, current_node] > 0
        Pa1 = Pa * (~cond_set)
        reachability_matrix[np.concatenate((Pa1, [False] * p)), current_node] = True
        reachability_matrix[np.concatenate(([False] * p, Pa1)), current_node] = True

        if is_ancestor_of_cond_set[current_node]:
            reachability_matrix[current_node, np.concatenate(([False] * p, Pa))] = True

            if path_matrix_2[i, current_node]:
                Pa_els = _list_of_index(Pa)
                if len(Pa_els) > 0:
                    reachable_on_non_causal_path_later = np.vstack(
                        [
                            reachable_on_non_causal_path_later,
                            np.column_stack([np.repeat(current_node, len(Pa_els)), Pa_els]),
                        ]
                    )

            new_nodes_to_check = _list_of_index(Pa * (~node_is_already_checked))
            queue_of_nodes_to_check.extend(new_nodes_to_check)

        if not cond_set[current_node]:
            reachability_matrix[current_node + p, np.concatenate(([False] * p, Pa))] = True
            new_nodes_to_check = _list_of_index(Pa * (~node_is_already_checked))
            queue_of_nodes_to_check.extend(new_nodes_to_check)

        Ch = G[current_node, :] > 0
        Ch1 = Ch * (~cond_set)
        reachability_matrix[np.concatenate(([False] * p, Ch1)), current_node + p] = True

        Ch2 = Ch * is_ancestor_of_cond_set
        reachability_matrix[np.concatenate((Ch2, [False] * p)), current_node + p] = True

        Ch2b = Ch2 * (path_matrix_2[i, :] > 0)
        Ch2b_els = _list_of_index(Ch2b)
        if len(Ch2b_els) > 0:
            reachable_on_non_causal_path_later = np.vstack(
                [
                    reachable_on_non_causal_path_later,
                    np.column_stack([Ch2b_els, np.repeat(current_node, len(Ch2b_els))]),
                ]
            )

        if not cond_set[current_node]:
            reachability_matrix[current_node, np.concatenate((Ch, [False] * p))] = True
            reachability_matrix[current_node + p, np.concatenate((Ch, [False] * p))] = True
            new_nodes_to_check = _list_of_index(Pa * (~node_is_already_checked))
            queue_of_nodes_to_check.extend(new_nodes_to_check)

    reachability_matrix = _compute_path_matrix(reachability_matrix)
    tt2 = np.sum(reachability_matrix[reachable_nodes, :], axis=0) > 0
    reachable_nodes[tt2] = True

    tt = np.sum(reachability_matrix[reachable_on_non_causal_path > 0, :], axis=0) > 0
    reachable_on_non_causal_path[tt] = True

    length_later = len(reachable_on_non_causal_path_later)
    if length_later > 2:
        for kk in range(2, length_later):
            reachable_through = int(reachable_on_non_causal_path_later[kk, 0])
            new_reachable = int(reachable_on_non_causal_path_later[kk, 1])
            reachable_on_non_causal_path[new_reachable + p] = True

            reachability_matrix[new_reachable, reachable_through] = False
            reachability_matrix[new_reachable, reachable_through + p] = False
            reachability_matrix[new_reachable + p, reachable_through] = False
            reachability_matrix[new_reachable + p, reachable_through + p] = False

        tt = np.sum(reachability_matrix[reachable_on_non_causal_path > 0, :], axis=0) > 0
        reachable_on_non_causal_path[tt] = True

    reachable_with_incoming_edge = reachable_on_non_causal_path[:p]
    reachable_with_outgoing_edge = reachable_on_non_causal_path[p:2 * p]

    return reachable_with_incoming_edge + reachable_with_outgoing_edge


def _compute_sid_row(G, i, parents_H_idx, path_matrix, parents_G):
    """
    Computes the mismatch vector of length p for source node i given estimated parents parents_H_idx.
    """
    p = len(G)
    parents_H = np.zeros(p, dtype=bool)
    if len(parents_H_idx) > 0:
        parents_H[list(parents_H_idx)] = True

    row_mismatch = np.zeros(p, dtype=bool)

    if np.array_equal(parents_G, parents_H):
        return row_mismatch

    path_matrix_2 = _compute_or_reuse_path_matrix(G, path_matrix, parents_H)
    reachability = _reachable_on_non_directed_path(G, i, parents_H, path_matrix, path_matrix_2)

    for j in range(p):
        if i == j:
            continue

        ij_g_null = not path_matrix[i, j]
        ij_h_null = bool(parents_H[j])

        if not ij_g_null and ij_h_null:
            row_mismatch[j] = True
            continue

        if ij_g_null and ij_h_null:
            continue

        if path_matrix[i, j]:
            children_on_directed_path = (G[i, :] > 0) * (path_matrix[:, j])
            if np.any(path_matrix[children_on_directed_path, :][:, parents_H]):
                row_mismatch[j] = True
                continue

        if reachability[j]:
            row_mismatch[j] = True
            continue

    return row_mismatch


def _get_possible_parent_sets(pdag, i_node, node_to_idx):
    """
    Finds all valid parent sets of node i_node in a PDAG/CPDAG.
    """
    directed_parents = {node_to_idx[n] for n in pdag.get_parents(i_node)}
    undirected_neighbors = {node_to_idx[n] for n in pdag.get_neighbors(i_node, edge_types={"--"})}

    if len(undirected_neighbors) == 0:
        return [frozenset(directed_parents)]

    possible_sets = []
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    for r in range(len(undirected_neighbors) + 1):
        for s_tuple in combinations(undirected_neighbors, r):
            s_set = set(s_tuple)

            # S must be a clique (every pair in S connected by undirected edge)
            is_clique = True
            for u, v in combinations(s_set, 2):
                if not pdag.has_edge(idx_to_node[u], idx_to_node[v], edge_type="--"):
                    is_clique = False
                    break
            if not is_clique:
                continue

            # No new unshielded v-structure: every u in S and v in directed_parents must be adjacent
            has_new_v = False
            for u in s_set:
                for v in directed_parents:
                    if not pdag.has_edge(idx_to_node[u], idx_to_node[v]):
                        has_new_v = True
                        break
                if has_new_v:
                    break
            if has_new_v:
                continue

            possible_sets.append(frozenset(directed_parents.union(s_set)))

    return possible_sets if len(possible_sets) > 0 else [frozenset(directed_parents)]


class SID(BaseSupervisedMetric):
    r"""
    Computes the Structural Intervention Distance (SID) between a true DAG and an estimated DAG or CPDAG (PDAG).

    The Structural Intervention Distance (Peters & Bühlmann, 2015) counts the number of interventional distributions
    :math:`P(Y_j \mid do(X_i = x_i))` that are incorrectly calculated when using the parent adjustment set from the
    estimated graph :math:`\mathcal{G}_{\text{est}}` on the true causal DAG :math:`\mathcal{G}_{\text{true}}`.

    Unlike Structural Hamming Distance (SHD), which measures purely graphical edge edits, SID evaluates the causal
    and interventional implications of the differences between the two graphs.

    Parameters
    ----------
    return_matrix: bool (default: False)
        If True, returns a boolean DataFrame matrix :math:`M` of shape :math:`(p, p)` where :math:`M[i, j] = 1`
        indicates that the interventional distribution :math:`P(Y_j \mid do(X_i))` is incorrectly estimated.
        For CPDAGs (PDAGs), returns the matrix according to `variant`.

    variant: str, optional (default: "mean")
        Specifies how to handle CPDAGs (PDAGs) which represent an equivalence class of DAGs:
        - ``"lower"``: Returns the lower bound (minimum errors over all DAG extensions).
        - ``"upper"``: Returns the upper bound (maximum errors over all DAG extensions).
        - ``"mean"``: Returns the average of lower and upper bounds.
        Ignored when both graphs are DAGs.

    Examples
    --------
    >>> from pgmpy.metrics import SID
    >>> from pgmpy.base import DAG, PDAG
    >>> dag1 = DAG([(1, 2), (1, 3), (2, 3)])
    >>> dag2 = DAG([(1, 3), (2, 3)])
    >>> sid = SID()
    >>> sid(true_causal_graph=dag1, est_causal_graph=dag2)
    2
    >>> sid(true_causal_graph=dag2, est_causal_graph=dag1)  # Asymmetric causal metric
    0

    Evaluating against CPDAGs (PDAGs):

    >>> true_chain = DAG([(1, 2), (2, 3)])
    >>> pdag = PDAG(edge_list=[(1, 2, "--"), (2, 3, "->")])
    >>> sid_lower = SID(variant="lower")
    >>> sid_lower(true_causal_graph=true_chain, est_causal_graph=pdag)
    0


    Returning the intervention mismatch matrix:

    >>> sid_mat = SID(return_matrix=True)
    >>> sid_mat(true_causal_graph=dag1, est_causal_graph=dag2)  # doctest: +NORMALIZE_WHITESPACE
           1      2      3
    1  False  False  False
    2   True  False   True
    3  False  False  False

    References
    ----------
    - Peters, J., & Bühlmann, P. (2015). Structural intervention distance (SID) for evaluating causal graphs.
      Neural Computation, 27(4), 771-799.
    """

    _tags = {
        "name": "SID",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
        "is_default": True,
    }

    def __init__(self, return_matrix: bool = False, variant: str = "mean"):
        if variant not in ("lower", "upper", "mean"):
            raise ValueError(f"variant must be one of 'lower', 'upper', or 'mean', got '{variant}'.")
        self.return_matrix = return_matrix
        self.variant = variant
        super().__init__()

    def _evaluate(self, true_causal_graph, est_causal_graph):
        if isinstance(true_causal_graph, PDAG) and len(true_causal_graph.undirected_edges) > 0:
            raise ValueError("true_causal_graph must be a fully directed DAG without undirected edges.")

        nodes_list = list(true_causal_graph.nodes())
        p = len(nodes_list)
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}

        # Binary adjacency matrix of true graph: G[i, j] = 1 if edge i -> j exists
        G = true_causal_graph.to_adjacency(encoding="binary", nodelist=nodes_list).to_numpy(copy=True)
        path_matrix = _compute_path_matrix(G)

        if isinstance(est_causal_graph, DAG):
            H = est_causal_graph.to_adjacency(encoding="binary", nodelist=nodes_list).to_numpy(copy=True)
            sid_mat = np.zeros((p, p), dtype=bool)

            for i in range(p):
                parents_G = G[:, i] > 0
                parents_H_idx = _list_of_index(H[:, i] > 0)
                sid_mat[i, :] = _compute_sid_row(G, i, parents_H_idx, path_matrix, parents_G)

            if self.return_matrix:
                return pd.DataFrame(sid_mat, index=nodes_list, columns=nodes_list)
            return int(np.sum(sid_mat))

        # est_causal_graph is a PDAG / CPDAG
        m_lower = np.zeros((p, p), dtype=bool)
        m_upper = np.zeros((p, p), dtype=bool)

        for i, i_node in enumerate(nodes_list):
            parents_G = G[:, i] > 0
            possible_parent_sets = _get_possible_parent_sets(est_causal_graph, i_node, node_to_idx)

            # Evaluate each possible parent set orientation
            evals = [_compute_sid_row(G, i, p_set, path_matrix, parents_G) for p_set in possible_parent_sets]
            eval_stack = np.vstack(evals)

            # Lower bound: error only if all parent extensions result in an error
            m_lower[i, :] = np.all(eval_stack, axis=0)
            # Upper bound: error if any parent extension results in an error
            m_upper[i, :] = np.any(eval_stack, axis=0)

        sid_lower = int(np.sum(m_lower))
        sid_upper = int(np.sum(m_upper))

        if self.return_matrix:
            if self.variant == "lower":
                return pd.DataFrame(m_lower, index=nodes_list, columns=nodes_list)
            elif self.variant == "upper":
                return pd.DataFrame(m_upper, index=nodes_list, columns=nodes_list)
            else:
                mean_mat = (m_lower.astype(float) + m_upper.astype(float)) / 2.0
                return pd.DataFrame(mean_mat, index=nodes_list, columns=nodes_list)

        if self.variant == "lower":
            return sid_lower
        elif self.variant == "upper":
            return sid_upper
        else:
            mean_val = (sid_lower + sid_upper) / 2.0
            return int(mean_val) if mean_val.is_integer() else mean_val
