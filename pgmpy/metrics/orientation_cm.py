from typing import List, Optional

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.metrics import _BaseSupervisedMetric


class OrientationConfusionMatrix(_BaseSupervisedMetric):
    """
    Computes confusion matrix based metrics for comparing edge orientations in DAGs.

    Treats edge direction as a binary classification problem, conditioned on edges
    that are present in both skeletons (common edges). Only supported for DAGs.

    Parameters
    ----------
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.

            cm          : Confusion matrix for edge direction among common skeleton edges.
            precision   : Fraction of estimated directed edges with correct orientation (TP / (TP + FP)).
            recall      : Fraction of true directed edges that are correctly oriented (TP / (TP + FN)).
            f1          : Harmonic mean of precision and recall.
            npv         : Fraction of absent estimated directions that are truly absent (TN / (TN + FN)).
            specificity : Fraction of truly absent directions correctly predicted absent (TN / (TN + FP)).

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.

    Examples
    --------
    >>> from pgmpy.metrics import OrientationConfusionMatrix
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
    >>> cm = OrientationConfusionMatrix()
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> result["precision"]
    1.0
    >>> result["recall"]
    1.0
    >>> result["cm"]  # doctest: +NORMALIZE_WHITESPACE
    Estimated       Est Present  Est Absent
    Actual
    Actual Present            2           0
    Actual Absent             0           2

    Compute only selected metrics:

    >>> cm = OrientationConfusionMatrix(metrics=["precision", "recall"])
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> "cm" in result
    False

    References
    ----------
    .. [1] Bryan Andrews, Joseph Ramsey, Gregory F. Cooper Proceedings of Machine Learning Research,
           PMLR 104:4-21, 2019. https://proceedings.mlr.press/v104/andrews19a.html
    """

    _tags = {
        "name": "orientation_confusion_matrix",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
    }

    def __init__(self, metrics: Optional[List[str]] = None):
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
        """Evaluate orientation confusion matrix metrics."""

        # Step 1: Get adjacency matrices for both graphs.
        nodes_list = sorted(true_causal_graph.nodes())
        true_adj = nx.adjacency_matrix(
            true_causal_graph, nodelist=nodes_list, weight=None
        ).todense()
        est_adj = nx.adjacency_matrix(
            est_causal_graph, nodelist=nodes_list, weight=None
        ).todense()

        true_skel = true_adj + true_adj.T
        est_skel = est_adj + est_adj.T
        common_edges = (true_skel > 0) & (est_skel > 0)

        # Step 2: Compute confusion matrix components.
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

        # Step 3: Compute specified metrics.
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
