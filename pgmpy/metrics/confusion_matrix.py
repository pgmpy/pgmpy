import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Union

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import _BaseSupervisedMetric


class ConfusionMatrix(_BaseSupervisedMetric):
    """
    Computes confusion matrix based metrics for comparing causal graphs.

    Implements adjacency and orientation confusion matrices with standard
    classification metrics (precision, recall, F1, NPV, specificity).

    Examples
    --------
    >>> from pgmpy.metrics import ConfusionMatrix
    >>> from pgmpy.base import DAG
    >>> true_dag = DAG([('A', 'B'), ('B', 'C'), ('A', 'C')])
    >>> est_dag = DAG([('A', 'B'), ('C', 'B')])
    >>> cm = ConfusionMatrix()
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> result['adjacency_precision']
    1.0
    """

    _tags = {
        "name": "ConfusionMatrix",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
    }

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize confusion matrix metrics calculator.

        Parameters
        ----------
        metrics : List[str], optional
            List of metrics to compute. If None, computes all available metrics.
            Available adjacency metrics: 'precision', 'recall', 'f1', 'npv', 'specificity'
            Available orientation metrics: 'orientation_precision', 'orientation_recall'
        """
        self.metrics = metrics or [
            'precision', 'recall', 'f1', 'npv', 'specificity',
            'orientation_precision', 'orientation_recall'
        ]
        super().__init__()

    def _get_skeleton_adjacency_matrix(self, graph: Union[DAG, PDAG]) -> np.ndarray:
        """Get undirected skeleton adjacency matrix."""
        nodes_list = sorted(list(graph.nodes()))

        if isinstance(graph, DAG):
            nx_graph = nx.DiGraph(graph.edges())
            nx_graph.add_nodes_from(nodes_list)
            adj_matrix = nx.adjacency_matrix(nx_graph, nodelist=nodes_list).todense()
            skeleton = adj_matrix + adj_matrix.T
            skeleton[skeleton > 0] = 1
            return skeleton
        else:  # PDAG
            nx_graph = nx.Graph()
            for edge in graph.edges():
                nx_graph.add_edge(edge[0], edge[1])
            nx_graph.add_nodes_from(nodes_list)
            return nx.adjacency_matrix(nx_graph, nodelist=nodes_list).todense()

    def _compute_adjacency_confusion_matrix(self, true_graph: Union[DAG, PDAG],
                                          est_graph: Union[DAG, PDAG]) -> Dict[str, int]:
        """Compute adjacency confusion matrix."""
        true_adj = self._get_skeleton_adjacency_matrix(true_graph)
        est_adj = self._get_skeleton_adjacency_matrix(est_graph)

        mask = np.triu(np.ones_like(true_adj, dtype=bool), k=1)

        true_edges = np.asarray(true_adj[mask]).flatten()
        est_edges = np.asarray(est_adj[mask]).flatten()

        tp = int(np.sum((true_edges == 1) & (est_edges == 1)))
        fp = int(np.sum((true_edges == 0) & (est_edges == 1)))
        fn = int(np.sum((true_edges == 1) & (est_edges == 0)))
        tn = int(np.sum((true_edges == 0) & (est_edges == 0)))

        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def _compute_orientation_confusion_matrix(self, true_graph: DAG,
                                            est_graph: DAG) -> Dict[str, int]:
        """Compute conditional orientation confusion matrix for correctly placed edges."""
        nodes_list = sorted(list(true_graph.nodes()))

        true_nx = nx.DiGraph(true_graph.edges())
        true_nx.add_nodes_from(nodes_list)
        true_adj = nx.adjacency_matrix(true_nx, nodelist=nodes_list).todense()

        est_nx = nx.DiGraph(est_graph.edges())
        est_nx.add_nodes_from(nodes_list)
        est_adj = nx.adjacency_matrix(est_nx, nodelist=nodes_list).todense()

        true_skel = true_adj + true_adj.T
        est_skel = est_adj + est_adj.T
        common_edges = (true_skel > 0) & (est_skel > 0)

        tp = fp = fn = tn = 0

        for i in range(true_adj.shape[0]):
            for j in range(true_adj.shape[1]):
                if common_edges[i, j] and i != j:
                    true_arrow = true_adj[i, j] == 1
                    est_arrow = est_adj[i, j] == 1

                    if true_arrow and est_arrow:
                        tp += 1
                    elif not true_arrow and not est_arrow:
                        tn += 1
                    elif not true_arrow and est_arrow:
                        fp += 1
                    elif true_arrow and not est_arrow:
                        fn += 1

        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def _evaluate(self, true_causal_graph, est_causal_graph):
        """Evaluate confusion matrix metrics."""
        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        results = {}

        # Adjacency confusion matrix
        adj_cm = self._compute_adjacency_confusion_matrix(true_causal_graph, est_causal_graph)
        results['adjacency_confusion_matrix'] = adj_cm

        tp, fp, fn, tn = adj_cm['tp'], adj_cm['fp'], adj_cm['fn'], adj_cm['tn']

        if 'precision' in self.metrics:
            results['adjacency_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if 'recall' in self.metrics:
            results['adjacency_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if 'f1' in self.metrics:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results['adjacency_f1'] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if 'npv' in self.metrics:
            results['adjacency_npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        if 'specificity' in self.metrics:
            results['adjacency_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Orientation metrics (only for DAGs)
        if isinstance(true_causal_graph, DAG) and isinstance(est_causal_graph, DAG):
            orient_cm = self._compute_orientation_confusion_matrix(true_causal_graph, est_causal_graph)
            results['orientation_confusion_matrix'] = orient_cm

            otp, ofp, ofn = orient_cm['tp'], orient_cm['fp'], orient_cm['fn']

            if 'orientation_precision' in self.metrics:
                results['orientation_precision'] = otp / (otp + ofp) if (otp + ofp) > 0 else 0.0

            if 'orientation_recall' in self.metrics:
                results['orientation_recall'] = otp / (otp + ofn) if (otp + ofn) > 0 else 0.0

        return results
