from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyConfusionMatrix


class AdjacencyRecall(AdjacencyConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencyRecall returns a scalar value denoting the fraction of true skeleton edges that are
    recovered i.e (TP / (TP + FN)).

    Returns
    -----
    float
        The Adjacency Recall value.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencyRecall
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
    >>> scorer = AdjacencyRecall()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    0.5
    """

    _tags = {
        "name": "adjacency_recall",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["recall"])
        scorer.evaluate(true_causal_graph, est_causal_graph)
        return scorer.recall_
