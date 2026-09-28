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
        cm = AdjacencyConfusionMatrix(metrics=["precision"])
        precision = cm.evaluate(true_causal_graph, est_causal_graph)["precision"]
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
        cm = AdjacencyConfusionMatrix(metrics=["recall"])
        recall = cm.evaluate(true_causal_graph, est_causal_graph)["recall"]
        return recall


class AdjacencyF1(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencyF1 returns a scalar value denoting the harmonic mean of both precision and recall.

    Returns
    -----
    float
        The Adjacency F1 Score.
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
        cm = AdjacencyConfusionMatrix(metrics=["f1"])
        f1 = cm.evaluate(true_causal_graph, est_causal_graph)["f1"]
        return f1


class AdjacencyNPV(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Adjacency Confusion Matrix,
    AdjacencyNPV returns a scalar value denoting the fraction of absent estimated edges that are truly abselt
    i.e (TN / (TN + FN)).

    Returns
    -----
    float
        The Adjacency Negative Predictive Value.

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
        cm = AdjacencyConfusionMatrix(metrics=["npv"])
        npv = cm.evaluate(true_causal_graph, est_causal_graph)["npv"]
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
        cm = AdjacencyConfusionMatrix(metrics=["specificity"])
        specificity = cm.evaluate(true_causal_graph, est_causal_graph)["specificity"]
        return specificity
