from pgmpy.base import DAG
from pgmpy.metrics import OrientationConfusionMatrix


class OrientationSpecificity(OrientationConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Orientation Confusion Matrix,
    OrientationSpecificity returns a scalar value denoting the fraction of estimated skeleton edges that are correct
    i.e (TP / (TP + FP)).

    Examples
    --------
    >>> from pgmpy.metrics import OrientationSpecificity
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
    >>> scorer = OrientationSpecificity()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0
    """

    _tags = {
        "name": "orientation_specificity",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": [DAG],
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["specificity"])
        scorer.evaluate(true_causal_graph, est_causal_graph)["specificity"]
        return scorer.specificity_
