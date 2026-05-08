import networkx as nx
import numpy as np

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import _BaseSupervisedMetric


class SHD(_BaseSupervisedMetric):
    r"""
    Computes the Structural Hamming Distance (SHD) between two graphs.

    Given two graphs (DAGs or PDAGs) :math:`G_1` and :math:`G_2` over the same vertex set, let :math:`S(G)` denote the
    skeleton (underlying undirected graph) of :math:`G`. The SHD is:

    .. math::

        \text{SHD}(G_1, G_2) = |S(G_1) \triangle S(G_2)|
            + p \cdot |\{e \in S(G_1) \cap S(G_2) :
            \text{orient}(e, G_1) \neq \text{orient}(e, G_2)\}|

    where :math:`\triangle` js the symmetric difference and :math:`p` is the ``edge_reverse_penalty`` (default 1; set to
    2 to count a reversal as one deletion plus one addition).

    For PDAGs, an undirected edge is represented as a pair of directed edges, so any orientation mismatch on a shared
    skeleton edge counts as one operation.

    Parameters
    ----------
    edge_reverse_penalty: int (default: 1)
        Penalty :math:`p` for orientation mismatches on shared skeleton edges.

    Examples
    --------
    >>> from pgmpy.metrics import SHD
    >>> from pgmpy.base import DAG
    >>> dag1 = DAG([(1, 2), (2, 3)])
    >>> dag2 = DAG([(2, 1), (2, 3)])
    >>> shd = SHD()
    >>> shd(true_causal_graph=dag1, est_causal_graph=dag2)
    1
    >>> shd_double = SHD(edge_reverse_penalty=2)
    >>> shd_double(true_causal_graph=dag1, est_causal_graph=dag2)
    2

    PDAGs are also supported:

    >>> from pgmpy.base import PDAG
    >>> pdag1 = PDAG(directed_ebunch=[(1, 2)], undirected_ebunch=[(2, 3)])
    >>> pdag2 = PDAG(directed_ebunch=[(1, 2), (2, 3)])
    >>> shd(true_causal_graph=pdag1, est_causal_graph=pdag2)
    1
    """

    _tags = {
        "name": "SHD",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": True,
        "supported_graph_types": (DAG, PDAG),
        "is_default": True,
    }

    def __init__(self, edge_reverse_penalty=1):
        if edge_reverse_penalty not in (1, 2):
            raise ValueError(f"edge_reverse_penalty must be 1 or 2, got '{edge_reverse_penalty}'.")
        self.edge_reverse_penalty = edge_reverse_penalty
        super().__init__()

    def _evaluate(self, true_causal_graph, est_causal_graph):

        nodes_list = true_causal_graph.nodes()

        dag_true = nx.DiGraph(true_causal_graph.edges())
        dag_true.add_nodes_from(list(nx.isolates(true_causal_graph)))
        m1 = nx.adjacency_matrix(dag_true, nodelist=nodes_list).todense()

        dag_est = nx.DiGraph(est_causal_graph.edges())
        dag_est.add_nodes_from(list(nx.isolates(est_causal_graph)))
        m2 = nx.adjacency_matrix(dag_est, nodelist=nodes_list).todense()

        shd = 0

        # Skeletons: 1 wherever there is any edge (directed or undirected) between a pair.
        # For PDAGs, undirected edges have both m[i, j] and m[j, i] set to 1, so m + m.T
        # produces a 2 at those positions; clipping to {0, 1} keeps the skeleton comparison
        # honest and avoids double-counting an undirected-vs-nothing difference.
        s1 = np.clip(m1 + m1.T, 0, 1)
        s2 = np.clip(m2 + m2.T, 0, 1)

        # Edges that are in m1 but not in m2 (deletions from m1)
        ds = s1 - s2
        ind = np.where(ds > 0)
        m1[ind] = 0
        shd = shd + (len(ind[0]) / 2)

        # Edges that are in m2 but not in m1 (additions to m1)
        ind = np.where(ds < 0)
        m1[ind] = m2[ind]
        shd = shd + (len(ind[0]) / 2)

        # Edges that need to be simply reversed
        d = np.abs(m1 - m2)
        reversal_count = np.sum((d + d.T) > 0) / 2
        shd = shd + (reversal_count * self.edge_reverse_penalty)

        return int(shd)
