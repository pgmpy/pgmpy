from pgmpy.base import DAG
from pgmpy.metrics import BaseSupervisedMetric, OrientationConfusionMatrix


class OrientationPrecision(BaseSupervisedMetric):
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
        precision = scorer.evaluate(true_causal_graph, est_causal_graph)["precision"]
        return precision


class OrientationRecall(BaseSupervisedMetric):
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
        recall = scorer.evaluate(true_causal_graph, est_causal_graph)["recall"]
        return recall


class OrientationF1Score(BaseSupervisedMetric):
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
        "supported_graph_types": (DAG,),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["f1"])
        f1 = scorer.evaluate(true_causal_graph, est_causal_graph)["f1"]
        return f1


class OrientationNPV(BaseSupervisedMetric):
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
        "supported_graph_types": (DAG,),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["npv"])
        npv = scorer.evaluate(true_causal_graph, est_causal_graph)["npv"]
        return npv


class OrientationSpecificity(BaseSupervisedMetric):
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
        "supported_graph_types": (DAG,),
        "output_type": "scalar",
    }

    def _evaluate(self, true_causal_graph, est_causal_graph):
        scorer = OrientationConfusionMatrix(metrics=["specificity"])
        specificity = scorer.evaluate(true_causal_graph, est_causal_graph)["specificity"]
        return specificity
