from pgmpy.base import DAG
from pgmpy.metrics import BaseSupervisedMetric, OrientationConfusionMatrix


class OrientationPrecision(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Orientation Confusion Matrix,
    OrientationPrecision returns a scalar value denoting the fraction of estimated directed edges that are
    correctly oriented i.e (TP / (TP + FP)).
    """
    _tags = {
    "name": "orientation_precision",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar"
    }
    def _evaluate(self, true_causal_graph, est_causal_graph):
        cm = OrientationConfusionMatrix(metrics=["precision"])
        precision = cm.evaluate(true_causal_graph, est_causal_graph)["precision"]
        return precision


class OrientationRecall(BaseSupervisedMetric):
    """ 
    Derived from the computation of true and false nodes used to build the Irientation Confusion Matrix,
    OrientationRecall returns a scalar value denoting the fraction of true directed edges that are correctly
    oriented i.e (TP / (TP + FN)).
    """
    _tags = {
        "name": "orientation_recall",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar"
    }
    def _evaluate(self, true_causal_graph, est_causal_graph):
        cm = OrientationConfusionMatrix(metrics=["recall"])
        recall = cm.evaluate(true_causal_graph, est_causal_graph)["recall"]
        return recall


class OrientationF1(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Irientation Confusion Matrix,
    OrientationF1 returns a scalar value denoting the harmonic mean of both precision and recall.
    """
    _tags = {
        "name": "orientation_f1",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar"
    }
    def _evaluate(self, true_causal_graph, est_causal_graph):
        cm = OrientationConfusionMatrix(metrics=["f1"])
        f1 = cm.evaluate(true_causal_graph, est_causal_graph)["f1"]
        return f1


class OrientationNPV(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Irientation Confusion Matrix,
    OrientationNPV returns a scalar value denoting the fraction of estimated skeleton edges that are correct
    i.e (TN / (TN + FN)).
    """
    _tags = {
        "name": "orientation_npv",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar"
    }
    def _evaluate(self, true_causal_graph, est_causal_graph):
        cm = OrientationConfusionMatrix(metrics=["npv"])
        npv = cm.evaluate(true_causal_graph, est_causal_graph)["npv"]
        return npv


class OrientationSpecificity(BaseSupervisedMetric):
    """
    Derived from the computation of true and false nodes used to build the Orientation Confusion Matrix,
    OrientationSpecificity returns a scalar value denoting the fraction of estimated skeleton edges that are correct
    i.e (TP / (TP + FP)).
    """
    _tags = {
        "name": "orientation_specificity",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
        "output_type": "scalar"
    }
    def _evaluate(self, true_causal_graph, est_causal_graph):
        cm = OrientationConfusionMatrix(metrics=["specificity"])
        specificity = cm.evaluate(true_causal_graph, est_causal_graph)["specificity"]
        return specificity
