from pgmpy.base import DAG
from pgmpy.metrics import OrientationConfusionMatrix


class OrientationF1Score(OrientationConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Irientation Confusion Matrix,
    OrientationF1 returns a scalar value denoting the harmonic mean of both precision and recall.

    Returns
    -----
    float
        The Orientation F1 Score.

    Examples
    --------
    >>> from pgmpy.metrics import OrientationF1Score
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
    >>> scorer = OrientationF1Score()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0
    """

    _tags = {
        "name": "orientation_f1",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": [DAG],
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["f1"])
        scorer.evaluate(true_causal_graph, est_causal_graph)["f1"]
        return scorer.f1_
