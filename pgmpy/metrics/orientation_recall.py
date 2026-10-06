from pgmpy.base import DAG
from pgmpy.metrics import OrientationConfusionMatrix


class OrientationRecall(OrientationConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Irientation Confusion Matrix,
    OrientationRecall returns a scalar value denoting the fraction of true directed edges that are correctly
    oriented i.e (TP / (TP + FN)).

    Returns
    -----
    float
        The Orientation Recall value.

    Examples
    --------
    >>> from pgmpy.metrics import OrientationRecall
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
    >>> scorer = OrientationRecall()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0
    """

    _tags = {
        "name": "orientation_recall",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["recall"])
        scorer.evaluate(true_causal_graph, est_causal_graph)["recall"]
        return scorer.recall_
