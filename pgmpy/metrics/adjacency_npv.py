from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyConfusionMatrix


class AdjacencyNPV(AdjacencyConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencyNPV returns a scalar value denoting the fraction of absent estimated edges that are truly absent
    i.e (TN / (TN + FN)).

    Returns
    -----
    float
        The Adjacency Negative Predictive Value.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencyNPV
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
    >>> scorer = AdjacencyNPV()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    0.5
    """

    _tags = {
        "name": "adjacency_npv",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": [DAG, PDAG],
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["npv"])
        scorer.evaluate(true_causal_graph, est_causal_graph)
        return scorer.npv_
