from pgmpy.base import DAG, PDAG
from pgmpy.metrics import AdjacencyConfusionMatrix, BaseSupervisedMetric


class AdjacencyPrecision(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencyPrecision returns a scalar value denoting the fraction of estimated skeleton edges that are correct
    i.e (TP / (TP + FP)).

    Returns
    -----
    float
        The Adjacency Precision value.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencyPrecision
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
    >>> scorer = AdjacencyPrecision()
    >>> result = scorer.evaluate(true_dag, est_dag)
    >>> result
    1.0

    """

    _tags = {
        "name": "adjacency_precision",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["precision"])
        precision = scorer.evaluate(true_causal_graph, est_causal_graph)["precision"]
        return precision


class AdjacencyRecall(BaseSupervisedMetric):
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
        recall = scorer.evaluate(true_causal_graph, est_causal_graph)["recall"]
        return recall


class AdjacencyF1Score(BaseSupervisedMetric):
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
    0.666666666
    """

    _tags = {
        "name": "adjacency_f1",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["f1"])
        f1 = scorer.evaluate(true_causal_graph, est_causal_graph)["f1"]
        return f1


class AdjacencyNPV(BaseSupervisedMetric):
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
    1.0
    """

    _tags = {
        "name": "adjacency_f1",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = AdjacencyConfusionMatrix(metrics=["npv"])
        npv = scorer.evaluate(true_causal_graph, est_causal_graph)["npv"]
        return npv


class AdjacencySpecificity(BaseSupervisedMetric):
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
    0.5
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
        specificity = scorer.evaluate(true_causal_graph, est_causal_graph)["specificity"]
        return specificity
