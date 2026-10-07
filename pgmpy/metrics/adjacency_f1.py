from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyConfusionMatrix


class AdjacencyF1Score(AdjacencyConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencyF1 returns a scalar value denoting the harmonic mean of both precision and recall.


    Returns
    -----
    float
        The Adjacency F1 Score.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencyF1Score
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
    >>> scorer = AdjacencyF1Score()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    0.6666666666666666
    """

    _tags = {
        "name": "adjacency_f1",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": [DAG, PDAG],
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["f1"])
        scorer.evaluate(true_causal_graph, est_causal_graph)
        return scorer.f1_
