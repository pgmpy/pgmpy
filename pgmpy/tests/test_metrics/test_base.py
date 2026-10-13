import pytest
from skbase.lookup import all_objects

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import (
    SHD,
    AdjacencyConfusionMatrix,
    BaseSupervisedMetric,
    BaseUnsupervisedMetric,
    CorrelationScore,
    FisherC,
    OrientationConfusionMatrix,
    get_metric,
)


def test_subclass_inherits_tags():
    class MySHD(SHD):
        _tags = {"name": "my_shd"}

    graph = DAG([("A", "B")])
    assert MySHD()(true_causal_graph=graph, est_causal_graph=graph) == 0


def test_filter_metrics_by_tags():
    metrics = all_objects(
        object_types=[BaseSupervisedMetric, BaseUnsupervisedMetric],
        package_name="pgmpy.metrics",
        return_names=False,
        filter_tags={"supported_graph_types": PDAG},
    )
    assert set(metrics) == {SHD, AdjacencyConfusionMatrix}


def test_get_metric():
    assert isinstance(get_metric("SHD"), SHD)
    assert isinstance(get_metric("orientation_confusion_matrix", requires_true_graph=True), OrientationConfusionMatrix)
    assert isinstance(get_metric(requires_true_graph=True), SHD)
    assert isinstance(get_metric(requires_true_graph=False), CorrelationScore)
    shd = SHD()
    assert get_metric(shd, requires_true_graph=True) is shd

    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("not_a_metric")
    with pytest.raises(ValueError, match="compares against data"):
        get_metric("correlation_score", requires_true_graph=True)
    with pytest.raises(ValueError, match="compares against a true graph"):
        get_metric(shd, requires_true_graph=False)
    with pytest.raises(ValueError, match="Cannot select a default metric"):
        get_metric()


def test_output_type_follows_hyperparameters():
    assert CorrelationScore().get_tag("output_type") == "scalar"
    assert CorrelationScore(return_summary=True).get_tag("output_type") == "dataframe"
    assert FisherC().get_tag("output_type") == "scalar"
    assert FisherC(compute_rmsea=True).get_tag("output_type") == "tuple"
    assert CorrelationScore.get_class_tag("output_type") == "scalar"
