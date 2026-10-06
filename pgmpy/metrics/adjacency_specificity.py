from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyConfusionMatrix


class AdjacencySpecificity(AdjacencyConfusionMatrix):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencySpecificity returns a scalar value denoting the fraction of absent estimated edges that are truly absem=nt
    i.e (TN / (TN + FP)).

    Returns
    -----
    float
        The Adjacency True Negative Rate.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencySpecificity
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
    >>> scorer = AdjacencySpecificity()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0
    """

    _tags = {
        "name": "adjacency_specificity",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["specificity"])
        scorer.evaluate(true_causal_graph, est_causal_graph)
        return scorer.specificity_
