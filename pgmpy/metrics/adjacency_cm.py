import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import _BaseSupervisedMetric


class AdjacencyConfusionMatrix(_BaseSupervisedMetric):
    """
    Computes confusion matrix based metrics for comparing causal graph skeletons.

    Treats edge presence/absence in the undirected skeleton as a binary classification
    problem and computes confusion matrix based metrics.

    Parameters
    ----------
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.

            cm          : Confusion matrix for skeleton edge presence.
            precision   : Fraction of estimated skeleton edges that are correct (TP / (TP + FP)).
            recall      : Fraction of true skeleton edges that are recovered (TP / (TP + FN)).
            f1          : Harmonic mean of precision and recall.
            npv         : Fraction of absent estimated edges that are truly absent (TN / (TN + FN)).
            specificity : Fraction of truly absent edges correctly predicted absent (TN / (TN + FP)).

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencyConfusionMatrix
    >>> from pgmpy.base import DAG
    >>> true_dag = DAG(
    ...     [
    ...         ("Smoking", "Lung_Cancer"),
    ...         ("Smoking", "Heart_Disease"),
    ...         ("Age", "Heart_Disease"),
    ...         ("Age", "Lung_Cancer"),
    ...     ]
    ... )
    >>> est_dag = DAG([("Smoking", "Lung_Cancer"), ("Age", "Heart_Disease")])
    >>> cm = AdjacencyConfusionMatrix()
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> result["precision"]
    1.0
    >>> result["recall"]
    0.5
    >>> result["cm"]  # doctest: +NORMALIZE_WHITESPACE
    Estimated       Est Present  Est Absent
    Actual
    Actual Present            2           2
    Actual Absent             0           2

    Compute only selected metrics:

    >>> cm = AdjacencyConfusionMatrix(metrics=["precision", "recall", "f1"])
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> "f1" in result
    True
    >>> "npv" in result
    False

    References
    ----------
    .. [1] Petersen, A. H. (2025). Are you doing better than random guessing? a call for using negative controls
           when evaluating causal discovery algorithms. Proceedings of the Forty-First Conference on Uncertainty
           in Artificial Intelligence. Rio de Janeiro, Brazil: JMLR.org. https://arxiv.org/abs/2412.10039

    """

    _tags = {
        "name": "adjacency_confusion_matrix",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
    }

    def __init__(self, metrics: list[str] | None = None):
        self.metrics = metrics or [
            "cm",
            "precision",
            "recall",
            "f1",
            "npv",
            "specificity",
        ]
        super().__init__()

    def _evaluate(self, true_causal_graph, est_causal_graph):
        """Evaluate adjacency confusion matrix metrics."""
        # Step 1: Get adjacency matrices for both graphs
        nodes_list = sorted(true_causal_graph.nodes())
        true_adj = nx.adjacency_matrix(true_causal_graph, nodelist=nodes_list, weight=None).todense()
        est_adj = nx.adjacency_matrix(est_causal_graph, nodelist=nodes_list, weight=None).todense()

        true_skel = (true_adj + true_adj.T) > 0
        est_skel = (est_adj + est_adj.T) > 0

        mask = np.triu(np.ones_like(true_skel, dtype=bool), k=1)
        true_edges = np.asarray(true_skel[mask]).flatten()
        est_edges = np.asarray(est_skel[mask]).flatten()

        # Step 2: Compute confusion matrix components
        tp = int(np.sum(true_edges & est_edges))
        fp = int(np.sum(~true_edges & est_edges))
        fn = int(np.sum(true_edges & ~est_edges))
        tn = int(np.sum(~true_edges & ~est_edges))

        # Step 3: Compute specified metrics
        results = {}
        if "cm" in self.metrics:
            results["cm"] = pd.DataFrame(
                [[tp, fn], [fp, tn]],
                index=pd.Index(["Actual Present", "Actual Absent"], name="Actual"),
                columns=pd.Index(["Est Present", "Est Absent"], name="Estimated"),
            )

        if "precision" in self.metrics:
            results["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if "recall" in self.metrics:
            results["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if "f1" in self.metrics:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results["f1"] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if "npv" in self.metrics:
            results["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        if "specificity" in self.metrics:
            results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return results
