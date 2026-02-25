"""
Precision, recall, and F1 metrics for causal graph evaluation.
"""

from pgmpy.base import DAG, PDAG


def precision_recall(true_causal_graph, est_causal_graph):
    """
    Computes precision, recall, and F1 for skeleton, directed edges,
    and v-structures between a true and estimated causal graph.

    Parameters
    ----------
    true_causal_graph : pgmpy.base.DAG or pgmpy.base.PDAG
        The ground truth causal graph.
    est_causal_graph : pgmpy.base.DAG or pgmpy.base.PDAG
        The estimated causal graph output by a causal discovery algorithm.

    Returns
    -------
    dict
        A nested dictionary with keys 'skeleton', 'directed', 'v_structure',
        each containing 'precision', 'recall', 'f1'.

    Raises
    ------
    ValueError
        If true_causal_graph and est_causal_graph do not have the same node set.
    TypeError
        If either argument is not a DAG or PDAG instance.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from pgmpy.metrics import precision_recall
    >>> true = DAG([("X", "Y"), ("Y", "Z")])
    >>> est = DAG([("X", "Y"), ("Z", "Y")])
    >>> precision_recall(true, est)
    {'skeleton':    {'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
     'directed':    {'precision': 0.5, 'recall': 0.5, 'f1': 0.5},
     'v_structure': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}

    References
    ----------
    .. [1] Tsamardinos et al. (2006). The max-min hill-climbing Bayesian network
           structure learning algorithm. Machine Learning, 65(1), 31-78.
    """
    # --- Input validation ---
    if not isinstance(true_causal_graph, (DAG, PDAG)) or not isinstance(
        est_causal_graph, (DAG, PDAG)
    ):
        raise TypeError(
            "true_causal_graph and est_causal_graph must be DAG or PDAG instances."
        )
    if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
        raise ValueError(
            "true_causal_graph and est_causal_graph must have the same node set. "
            f"Got {set(true_causal_graph.nodes())} and {set(est_causal_graph.nodes())}."
        )

    def _prf(tp, fp, fn):
        """Compute precision, recall, F1 from counts. Returns 0.0 on zero denominator."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    # --- Skeleton ---
    # For PDAG: undirected edges contribute to skeleton.
    # Convert all edges to frozensets to ignore direction.
    if isinstance(true_causal_graph, PDAG):
        true_skel = {frozenset(e) for e in true_causal_graph.directed_edges} | {
            frozenset(e) for e in true_causal_graph.undirected_edges
        }
    else:
        true_skel = {frozenset(e) for e in true_causal_graph.edges()}

    if isinstance(est_causal_graph, PDAG):
        est_skel = {frozenset(e) for e in est_causal_graph.directed_edges} | {
            frozenset(e) for e in est_causal_graph.undirected_edges
        }
    else:
        est_skel = {frozenset(e) for e in est_causal_graph.edges()}

    skel_tp = len(true_skel & est_skel)
    skel_fp = len(est_skel - true_skel)
    skel_fn = len(true_skel - est_skel)
    skeleton_metrics = _prf(skel_tp, skel_fp, skel_fn)

    # --- Directed edges ---
    # For PDAG: only directed_edges count; undirected edges are excluded.
    if isinstance(true_causal_graph, PDAG):
        true_directed = set(true_causal_graph.directed_edges)
    else:
        true_directed = set(true_causal_graph.edges())

    if isinstance(est_causal_graph, PDAG):
        est_directed = set(est_causal_graph.directed_edges)
    else:
        est_directed = set(est_causal_graph.edges())

    dir_tp = len(true_directed & est_directed)
    dir_fp = len(est_directed - true_directed)
    dir_fn = len(true_directed - est_directed)
    directed_metrics = _prf(dir_tp, dir_fp, dir_fn)

    # --- V-structures ---
    # get_immoralities() returns {collider: [(sorted_p1, sorted_p2), ...]}
    # Flatten to a set of (collider, (p1, p2)) tuples for comparison.
    def _get_vstructs(model):
        if isinstance(model, PDAG):
            # PDAG doesn't have get_immoralities; convert to set of (collider, pair)
            # by checking directed predecessors only
            vstructs = set()
            import itertools

            for node in model.nodes():
                preds = [u for u in model.nodes() if model.has_directed_edge(u, node)]
                for p1, p2 in itertools.combinations(preds, 2):
                    if (
                        not model.has_directed_edge(p1, p2)
                        and not model.has_directed_edge(p2, p1)
                        and not model.has_undirected_edge(p1, p2)
                    ):
                        vstructs.add((node, tuple(sorted([p1, p2]))))
            return vstructs
        else:
            immoralities = model.get_immoralities()
            return {
                (collider, pair)
                for collider, pairs in immoralities.items()
                for pair in pairs
            }

    true_vstructs = _get_vstructs(true_causal_graph)
    est_vstructs = _get_vstructs(est_causal_graph)

    vs_tp = len(true_vstructs & est_vstructs)
    vs_fp = len(est_vstructs - true_vstructs)
    vs_fn = len(true_vstructs - est_vstructs)
    vstructure_metrics = _prf(vs_tp, vs_fp, vs_fn)

    return {
        "skeleton": skeleton_metrics,
        "directed": directed_metrics,
        "v_structure": vstructure_metrics,
    }
