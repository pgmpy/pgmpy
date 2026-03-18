import networkx as nx
import numpy as np

from pgmpy.base import DAG
from pgmpy.metrics import _BaseSupervisedMetric


class SHD(_BaseSupervisedMetric):
    """
    Computes the Structural Hamming Distance between `true_causal_graph` and `est_causal_graph`.

    SHD is defined as total number of basic operations: adding edges, removing edges, and reversing edges required to
    transform one graph to the other. It is a symmetrical measure.

    The code first accounts for edges that need to be deleted (from true_model), added (to true_model) and finally edges
    that need to be reversed. All operations count as 1.

    Examples
    --------
    >>> from pgmpy.metrics import SHD
    >>> from pgmpy.base import DAG
    >>> dag1 = DAG([(1, 2), (2, 3)])
    >>> dag2 = DAG([(2, 1), (2, 3)])
    >>> shd = SHD()
    >>> shd(true_causal_graph=dag1, est_causal_graph=dag2)
    1
    """

    _tags = {
        "name": "SHD",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": True,
        "is_symmetric": True,
        "supported_graph_types": (DAG,),
        "is_default": True,
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        nodes_list = true_causal_graph.nodes()

        dag_true = nx.DiGraph(true_causal_graph.edges())
        dag_true.add_nodes_from(list(nx.isolates(true_causal_graph)))
        m1 = nx.adjacency_matrix(dag_true, nodelist=nodes_list).todense()

        dag_est = nx.DiGraph(est_causal_graph.edges())
        dag_est.add_nodes_from(list(nx.isolates(est_causal_graph)))
        m2 = nx.adjacency_matrix(dag_est, nodelist=nodes_list).todense()

        shd = 0

        s1 = m1 + m1.T
        s2 = m2 + m2.T

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
        shd = shd + (np.sum((d + d.T) > 0) / 2)

        return int(shd)
