"""
Structural Intervention Distance (SID) for causal graph evaluation.
"""

import networkx as nx

from pgmpy.base import DAG
from pgmpy.metrics import _BaseSupervisedMetric


class SID(_BaseSupervisedMetric):
    """
    Computes the Structural Intervention Distance (SID) between
    `true_causal_graph` and `est_causal_graph`.

    SID counts the number of ordered pairs (i, j), i ≠ j, for which the
    interventional distribution P(Xj | do(Xi)) cannot be correctly identified
    using the estimated graph. For each pair, the parents of i in the true graph
    are used as the candidate adjustment set, and the backdoor criterion is
    checked in the estimated graph.

    Unlike SHD, SID is not symmetric: SID(G*, Ĝ) ≠ SID(Ĝ, G*) in general.
    Lower SID is better. The minimum value is 0 (perfect interventional
    equivalence) and the maximum is n*(n-1) for n nodes.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from pgmpy.metrics import SID
    >>> from pgmpy.base import DAG
    >>> true_dag = DAG([(1, 2), (2, 3)])
    >>> est_dag = DAG([(2, 1), (2, 3)])
    >>> sid = SID()
    >>> sid(true_causal_graph=true_dag, est_causal_graph=est_dag)
    4

    References
    ----------
    .. [1] Peters, J., & Bühlmann, P. (2015). Structural intervention distance
           for evaluating causal graphs. Neural Computation, 27(3), 771–799.
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

    def _evaluate(self, true_causal_graph, est_causal_graph):
        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        nodes = list(true_causal_graph.nodes())
        sid = 0

        for i in nodes:
            # Parents of i in true graph — always a valid backdoor set in true graph
            parents_i = set(true_causal_graph.predecessors(i))

            # Build modified estimated graph: remove all edges OUT of i
            # This is the graph we use to check backdoor path blocking
            est_modified = est_causal_graph.copy()
            edges_out_of_i = list(est_causal_graph.successors(i))
            est_modified.remove_edges_from([(i, child) for child in edges_out_of_i])

            # Descendants of i in the estimated graph (for condition A)
            desc_i_est = nx.descendants(est_causal_graph, i)

            for j in nodes:
                if i == j:
                    continue

                # Condition A: no node in parents_i is a descendant of i in est
                if parents_i & desc_i_est:
                    sid += 1
                    continue

                # Condition B: parents_i d-separates i from j in est_modified
                # i.e., i and j are NOT d-connected given parents_i in est_modified
                # Use is_dconnected on the modified graph (pgmpy DAG method)
                if est_modified.is_dconnected(i, j, observed=list(parents_i)):
                    sid += 1

        return sid
