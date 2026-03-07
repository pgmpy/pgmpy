import itertools

import networkx as nx
import numpy as np

from pgmpy.base import DAG, PDAG
from pgmpy.metrics._base import _BaseSupervisedMetric


class SHD(_BaseSupervisedMetric):
    """
    Computes the Structural Hamming Distance between `true_causal_graph` and `est_causal_graph`.

    SHD is defined as the total number of basic operations (adding, removing, or reversing
    edges) required to transform one graph into the other. It is a symmetrical measure.

    When both inputs are DAGs, the standard edge-level comparison is used. When either
    input is a PDAG, both graphs are automatically converted to their canonical CPDAG
    representations before comparison, making the metric Markov equivalence-aware: two
    DAGs in the same Markov Equivalence Class will score a distance of 0.

    Examples
    --------
    Standard SHD between two DAGs:

    >>> from pgmpy.metrics import SHD
    >>> from pgmpy.base import DAG
    >>> dag1 = DAG([(1, 2), (2, 3)])
    >>> dag2 = DAG([(2, 1), (2, 3)])
    >>> shd = SHD()
    >>> shd(true_causal_graph=dag1, est_causal_graph=dag2)
    1

    SHD between PDAG inputs (automatically CPDAG-aware):

    >>> from pgmpy.base import PDAG
    >>> chain = DAG([("X", "Y"), ("Y", "Z")])
    >>> fork = DAG([("Y", "X"), ("Y", "Z")])
    >>> shd(true_causal_graph=chain, est_causal_graph=fork)
    1
    >>> pdag1 = PDAG([("X", "Y"), ("Y", "Z")], [])
    >>> pdag2 = PDAG([("X", "Y")], [("Y", "Z")])
    >>> shd(true_causal_graph=pdag1, est_causal_graph=pdag2)
    0

    References
    ----------
    .. [1] Chickering, D. M. (2002). Optimal structure identification with
           greedy search. Journal of Machine Learning Research, 3, 507–554.
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

    def _evaluate(self, true_causal_graph, est_causal_graph):
        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        if isinstance(true_causal_graph, DAG) and isinstance(est_causal_graph, DAG):
            return self._standard_shd(true_causal_graph, est_causal_graph)
        return self._cpdag_shd(true_causal_graph, est_causal_graph)

    def _standard_shd(self, true_causal_graph, est_causal_graph):
        """Compute the standard SHD between two DAGs using adjacency matrices."""
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

    def _cpdag_shd(self, true_causal_graph, est_causal_graph):
        """Compute SHD at the CPDAG level for PDAG inputs."""

        def to_cpdag(g):
            if isinstance(g, DAG):
                return g.to_pdag()
            # PDAG input: canonicalize via Meek's rules so partially oriented
            # graphs in the same MEC become identical CPDAGs.
            return g.apply_meeks_rules(apply_r4=True, inplace=False)

        def edge_type(g, u, v):
            if g.has_directed_edge(u, v):
                return ("directed", u, v)
            elif g.has_directed_edge(v, u):
                return ("directed", v, u)
            elif g.has_undirected_edge(u, v):
                return ("undirected", frozenset([u, v]))
            return ("none", None)

        true_cpdag = to_cpdag(true_causal_graph)
        est_cpdag = to_cpdag(est_causal_graph)
        nodes = list(true_cpdag.nodes())

        return int(
            sum(
                edge_type(true_cpdag, u, v) != edge_type(est_cpdag, u, v)
                for u, v in itertools.combinations(nodes, 2)
            )
        )
