import itertools

from pgmpy.base import DAG, PDAG
from pgmpy.metrics._base import _BaseSupervisedMetric


class CPDAGSHD(_BaseSupervisedMetric):
    """
    Computes the Structural Hamming Distance between `true_causal_graph` and
    `est_causal_graph` at the level of their CPDAG representations.

    Standard SHD penalizes edges that are reversed but Markov-equivalent, i.e.,
    the two DAGs belong to the same Markov Equivalence Class (MEC) and encode
    the same conditional independencies. CPDAGSHD fixes this by converting both
    graphs to their unique CPDAG (Completed Partially Directed Acyclic Graph)
    before comparison, so that any two Markov-equivalent DAGs score 0.

    For each unordered node pair ``{u, v}``, the edge is classified as one of
    four types: no edge, undirected, ``u→v``, or ``v→u``. The distance equals
    the number of node pairs where the edge type differs between the two CPDAGs.

    Both DAG and PDAG inputs are accepted. A DAG input is converted to its CPDAG
    via :meth:`pgmpy.base.DAG.to_pdag` before comparison. A PDAG input (e.g.,
    the default output of PC or GES) is used directly.

    Parameters
    ----------
    true_causal_graph : pgmpy.base.DAG or pgmpy.base.PDAG
        The ground-truth causal graph.

    est_causal_graph : pgmpy.base.DAG or pgmpy.base.PDAG
        The estimated causal graph to evaluate.

    Returns
    -------
    int
        The CPDAG-level structural Hamming distance. When both inputs are
        DAGs, the distance is zero if and only if they belong to the same
        Markov equivalence class. For PDAG inputs, the comparison is
        performed directly without further canonicalization. The maximum
        value is ``n * (n - 1) / 2`` where ``n`` is the number of nodes.

    Raises
    ------
    ValueError
        If ``true_causal_graph`` and ``est_causal_graph`` do not have the
        same node set.

    Examples
    --------
    >>> from pgmpy.metrics import CPDAGSHD
    >>> from pgmpy.base import DAG
    >>> chain = DAG([("X", "Y"), ("Y", "Z")])
    >>> fork = DAG([("Y", "X"), ("Y", "Z")])
    >>> metric = CPDAGSHD()
    >>> metric(true_causal_graph=chain, est_causal_graph=fork)
    0
    >>> collider = DAG([("X", "Z"), ("Y", "Z")])
    >>> metric(true_causal_graph=chain, est_causal_graph=collider)
    3

    References
    ----------
    .. [1] Chickering, D. M. (2002). Optimal structure identification with
           greedy search. Journal of Machine Learning Research, 3, 507–554.
    """

    _tags = {
        "name": "CPDAGSHD",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": True,
        "supported_graph_types": (DAG, PDAG),
        "is_default": False,
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        def to_cpdag(g):
            if isinstance(g, DAG):
                return g.to_pdag()
            # If already a PDAG, force canonicalization using Meek's rules
            # so partially oriented graphs in the same MEC become identical CPDAGs.
            return g.apply_meeks_rules(apply_r4=True, inplace=False)

        def edge_type(g, u, v):
            if g.has_directed_edge(u, v):
                return (1, u, v)
            elif g.has_directed_edge(v, u):
                return (1, v, u)
            elif g.has_undirected_edge(u, v):
                return (0, 0, 0)
            return (-1, -1, -1)

        true_cpdag = to_cpdag(true_causal_graph)
        est_cpdag = to_cpdag(est_causal_graph)
        nodes = list(true_cpdag.nodes())

        return int(
            sum(
                edge_type(true_cpdag, u, v) != edge_type(est_cpdag, u, v)
                for u, v in itertools.combinations(nodes, 2)
            )
        )
