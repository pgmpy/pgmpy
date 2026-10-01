from pgmpy.base import DAG, PDAG
from pgmpy.metrics import (
    SHD,
    AdjacencyConfusionMatrix,
    CorrelationScore,
    FisherC,
    OrientationConfusionMatrix,
    get_metrics,
)


def test_subclass_inherits_tags():
    class MySHD(SHD):
        _tags = {"name": "my_shd"}

    graph = DAG([("A", "B")])
    assert MySHD()(true_causal_graph=graph, est_causal_graph=graph) == 0


def test_get_metrics_filters():
    assert set(get_metrics(supported_graph_types=PDAG)) == {SHD, AdjacencyConfusionMatrix}
    assert OrientationConfusionMatrix in get_metrics(requires_true_graph=True)
    assert get_metrics(name="SHD") == get_metrics(name="shd") == [SHD]
    assert get_metrics(default_for="supervised") == [SHD]
    assert get_metrics(default_for="unsupervised") == [CorrelationScore]


def test_output_type_follows_hyperparameters():
    assert CorrelationScore().get_tag("output_type") == "scalar"
    assert CorrelationScore(return_summary=True).get_tag("output_type") == "dataframe"
    assert FisherC().get_tag("output_type") == "scalar"
    assert FisherC(compute_rmsea=True).get_tag("output_type") == "tuple"
    assert CorrelationScore.get_class_tag("output_type") == "scalar"
