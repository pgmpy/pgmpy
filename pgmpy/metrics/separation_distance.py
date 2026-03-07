from itertools import combinations

from pgmpy.base import DAG
from pgmpy.metrics import _BaseSupervisedMetric


class SeparationDistance(_BaseSupervisedMetric):
    """
    Computes the Separation Distance (SD) between `true_causal_graph` and `est_causal_graph`.

    For every non-adjacent node pair in the estimated graph, a separating set is chosen according to a separation
    strategy (parents or ancestors). The metric then checks how many of those separating sets fail to d-separate
    the same pair in the true graph. The count is normalized by the total number of ordered node pairs.

    Unlike the SHD, which only considers individual edge differences, the separation distance captures whether two
    graphs agree on the d-separation statements they encode. This is especially useful when evaluating causal
    discovery methods whose output is defined up to Markov equivalence.

    The metric is asymmetric by default: separators are read from the estimated graph and verified against the
    true graph. A symmetric variant that averages both directions is available.

    Parameters
    ----------
    strategy: str (default: "parent")
        Separation strategy for choosing the separating set of a non-adjacent pair (u, v).

        - ``"parent"``: Uses pa(u) | pa(v), the union of parents.
        - ``"ancestor"``: Uses an({u, v}) \\ {u, v}, all ancestors excluding u and v.

    symmetric: bool (default: False)
        If True, returns the average of d(true, est) and d(est, true).

    Returns
    -------
    float
        Normalized separation distance in [0, 1]. Lower is better.

    Examples
    --------
    >>> from pgmpy.metrics import SeparationDistance
    >>> from pgmpy.base import DAG
    >>> dag1 = DAG([("A", "B"), ("B", "C")])
    >>> dag2 = DAG([("A", "B"), ("A", "C")])
    >>> sd = SeparationDistance(strategy="parent")
    >>> sd(true_causal_graph=dag1, est_causal_graph=dag2)
    0.0
    >>> sd(true_causal_graph=dag2, est_causal_graph=dag1)
    0.16666666666666666
    >>> sd_sym = SeparationDistance(strategy="parent", symmetric=True)
    >>> sd_sym(true_causal_graph=dag1, est_causal_graph=dag2)
    0.08333333333333333

    References
    ----------
    Wahl, J. and Runge, J. (2025). Separation-based distance measures for causal graphs.
    In Proc. 28th International Conference on AI and Statistics (AISTATS). arXiv:2402.04952.
    """

    _tags = {
        "name": "separation_distance",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
    }

    _STRATEGIES = ("parent", "ancestor")

    def __init__(self, strategy="parent", symmetric=False):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._STRATEGIES}, got '{strategy}'."
            )
        self.strategy = strategy
        self.symmetric = symmetric

    def _evaluate(self, true_causal_graph, est_causal_graph):
        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        num_nodes = len(true_causal_graph.nodes())
        if num_nodes < 2:
            return 0.0

        normalizer = num_nodes * (num_nodes - 1)

        dist_forward = self._one_direction_distance(
            base_graph=true_causal_graph,
            ref_graph=est_causal_graph,
            normalizer=normalizer,
        )

        if not self.symmetric:
            return dist_forward

        dist_backward = self._one_direction_distance(
            base_graph=est_causal_graph,
            ref_graph=true_causal_graph,
            normalizer=normalizer,
        )
        return (dist_forward + dist_backward) / 2.0

    def _one_direction_distance(self, base_graph, ref_graph, normalizer):
        """
        Computes the one-directional separation distance.

        For every non-adjacent pair in `ref_graph`, picks a separating set using
        the chosen strategy and checks whether it still d-separates the pair in
        `base_graph`.
        """
        penalty = 0

        for u, v in combinations(ref_graph.nodes(), 2):
            # Only consider non-adjacent pairs in the reference graph
            if ref_graph.has_edge(u, v) or ref_graph.has_edge(v, u):
                continue

            sep_set = self._get_separator(ref_graph, u, v)

            # Penalty when the separator no longer blocks all paths
            if base_graph.is_dconnected(u, v, observed=sep_set):
                penalty += 1

        return penalty / normalizer

    def _get_separator(self, graph, u, v):
        """
        Returns the separating set for non-adjacent pair (u, v) under the
        current strategy.
        """
        if self.strategy == "parent":
            sep_set = set(graph.get_parents(u)) | set(graph.get_parents(v))
        elif self.strategy == "ancestor":
            sep_set = graph.get_ancestors([u, v]) - {u, v}

        return list(sep_set)
