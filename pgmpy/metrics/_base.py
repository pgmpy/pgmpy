import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects


class BaseSupervisedMetric(BaseObject):
    """
    Base class for all metric classes in pgmpy that require ground truth causal graph.

    Every supervised metric sets all of the tags listed by ``pgmpy.registry.all_tags("supervised_metric")``. See
    :doc:`/api/tags` for what each tag means.
    """

    _tags = {
        "object_type": "supervised_metric",
        "name": None,
        "default_for": None,
        "requires_data": False,
        "is_symmetric": None,
        "requires_true_graph": True,
        "supported_graph_types": None,
        "output_type": None,
        "lower_is_better": None,
    }

    def evaluate(self, true_causal_graph, est_causal_graph, **kwargs):
        """
        Evaluate the metric by comparing the true causal graph with the estimated causal graph.

        Parameters
        ----------
        true_causal_graph: Instance of type pgmpy.base
            The ground truth causal graph.

        est_causal_graph: Instance of type pgmpy.base
            The estimated causal graph.
        """
        supported_graph_types = tuple(self.get_tag("supported_graph_types"))
        if not isinstance(true_causal_graph, supported_graph_types):
            raise ValueError(
                f"The true_causal_graph must be one of the following types: {supported_graph_types}, "
                f"but got {type(true_causal_graph)} instead."
            )

        if not isinstance(est_causal_graph, supported_graph_types):
            raise ValueError(
                f"The est_causal_graph must be one of the following types: {supported_graph_types}, "
                f"but got {type(est_causal_graph)} instead."
            )

        if not set(true_causal_graph.nodes()) == set(est_causal_graph.nodes()):
            raise ValueError("The `true_causal_graph` and `est_causal_graph` must be on the same nodes.")

        return self._evaluate(
            true_causal_graph=true_causal_graph,
            est_causal_graph=est_causal_graph,
            **kwargs,
        )

    def __call__(self, true_causal_graph, est_causal_graph, **kwargs):
        return self.evaluate(
            true_causal_graph=true_causal_graph,
            est_causal_graph=est_causal_graph,
            **kwargs,
        )


class BaseUnsupervisedMetric(BaseObject):
    """
    Base class for all metric classes in pgmpy that do not require ground truth causal graph.

    Every unsupervised metric sets all of the tags listed by ``pgmpy.registry.all_tags("unsupervised_metric")``. See
    :doc:`/api/tags` for what each tag means.
    """

    _tags = {
        "object_type": "unsupervised_metric",
        "name": None,
        "default_for": None,
        "requires_data": True,
        "requires_true_graph": False,
        "supported_graph_types": None,
        "output_type": None,
        "lower_is_better": None,
    }

    def evaluate(self, X, causal_graph, **kwargs):
        """
        Evaluate the metric by comparing the causal graph with the data.

        Parameters
        ----------
        X: pandas.DataFrame
            The data used for evaluation.

        causal_graph: Instance of type pgmpy.base
            The causal graph to be evaluated.
        """
        supported_graph_types = tuple(self.get_tag("supported_graph_types"))
        if not isinstance(causal_graph, supported_graph_types):
            raise ValueError(
                f"The causal_graph must be one of the following types: {supported_graph_types}, "
                f"but got {type(causal_graph)} instead."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"The data must be a pandas.DataFrame instance, but got {type(X)} instead.")
        elif len(set(causal_graph.nodes()) - set(X.columns)) > 0:
            raise ValueError(
                "Missing columns in data. Can't find values for the following variables: "
                f" {set(causal_graph.nodes()) - set(X.columns)}"
            )

        return self._evaluate(X=X, causal_graph=causal_graph, **kwargs)

    def __call__(self, X, causal_graph, **kwargs):
        return self.evaluate(X=X, causal_graph=causal_graph, **kwargs)


def get_metric(
    metric: str | BaseSupervisedMetric | BaseUnsupervisedMetric | None = None,
    requires_true_graph: bool | None = None,
) -> BaseSupervisedMetric | BaseUnsupervisedMetric:
    """
    Return a metric instance given a metric name, instance, or the kind of metric needed.

    Parameters
    ----------
    metric : str, BaseSupervisedMetric, BaseUnsupervisedMetric, or None, default=None
        The metric to return. A string selects the metric whose ``name`` tag matches (case-insensitive), instantiated
        with default parameters. A metric instance is returned unchanged. If ``None``, the default metric for
        ``requires_true_graph`` is returned.

    requires_true_graph : bool or None, default=None
        Whether the metric must compare against a true graph (``True``) or against data (``False``). Used to select
        the default metric when ``metric`` is ``None``, and to check the kind of the returned metric otherwise. If
        ``None``, the kind isn't checked.

    Returns
    -------
    BaseSupervisedMetric or BaseUnsupervisedMetric
        A metric instance ready to call.

    Raises
    ------
    ValueError
        If ``metric`` is an unknown name or not a string, metric instance, or ``None``; if both ``metric`` and
        ``requires_true_graph`` are ``None``; or if the metric's kind doesn't match ``requires_true_graph``.

    Examples
    --------
    >>> from pgmpy.metrics import get_metric
    >>> get_metric("SHD")
    SHD()
    >>> get_metric(requires_true_graph=False)
    CorrelationScore()
    """
    if metric is None or isinstance(metric, str):
        if metric is None:
            if requires_true_graph is None:
                raise ValueError("Cannot select a default metric: both `metric` and `requires_true_graph` are None.")
            filter_tags = {"default_for": "supervised" if requires_true_graph else "unsupervised"}
        else:
            filter_tags = {"name": metric.lower()}

        metric_classes = all_objects(
            object_types=[BaseSupervisedMetric, BaseUnsupervisedMetric],
            package_name="pgmpy.metrics",
            return_names=False,
            filter_tags=filter_tags,
        )
        if not metric_classes:
            raise ValueError(f"Unknown metric: {metric!r}.")
        metric = metric_classes[0]()
    elif not isinstance(metric, (BaseSupervisedMetric, BaseUnsupervisedMetric)):
        raise ValueError(f"Invalid `metric` argument: {metric!r}. Pass a metric name, a metric instance, or None.")

    compares_against = {True: "a true graph", False: "data"}
    if requires_true_graph is not None and metric.get_tag("requires_true_graph") != requires_true_graph:
        raise ValueError(
            f"Metric {metric.get_tag('name')!r} compares against {compares_against[not requires_true_graph]}, but a "
            f"metric that compares against {compares_against[requires_true_graph]} is required."
        )
    return metric
