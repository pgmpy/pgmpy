from pgmpy.base import DAG
from pgmpy.metrics import OrientationConfusionMatrix


class OrientationPrecision(OrientationConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Orientation Confusion Matrix,
    OrientationPrecision returns a scalar value denoting the fraction of estimated directed edges that are
    correctly oriented i.e (TP / (TP + FP)).

    Returns
    -----
    float
        The Orientation Precision metric.

    Examples
    --------
    >>> from pgmpy.metrics import OrientationPrecision
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
    >>> scorer = OrientationPrecision()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0
    """

    _tags = {
        "name": "orientation_precision",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["precision"])
        scorer.evaluate(true_causal_graph, est_causal_graph)["precision"]
        return scorer.precision_
