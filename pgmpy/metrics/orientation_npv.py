from pgmpy.base import DAG
from pgmpy.metrics import OrientationConfusionMatrix


class OrientationNPV(OrientationConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Irientation Confusion Matrix,
    OrientationNPV returns a scalar value denoting the fraction of estimated skeleton edges that are correct
    i.e (TN / (TN + FN)).

    Returns
    -----
    float
        The Orientation Negative Predictive Value.

    Examples
    --------
    >>> from pgmpy.metrics import OrientationNPV
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
    >>> scorer = OrientationNPV()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0
    """

    _tags = {
        "name": "orientation_npv",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": [DAG],
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["npv"])
        scorer.evaluate(true_causal_graph, est_causal_graph)["npv"]
        return scorer.npv_
