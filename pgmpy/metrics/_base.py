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


def get_metrics(**kwargs) -> list[type]:
    """
    Get metric classes matching the given tag filters.

    Parameters
    ----------
    **kwargs
        Tag filters passed to :func:`skbase.lookup.all_objects` via its ``filter_tags`` parameter. See
        :doc:`/api/tags` for the tags of metrics. ``name`` is matched case-insensitively.

    Returns
    -------
    list[type]
        Metric classes matching all the given tag filters. Empty if none match.

    Examples
    --------
    >>> from pgmpy.metrics import get_metrics
    >>> get_metrics(default_for="supervised")
    [<class 'pgmpy.metrics.shd.SHD'>]
    >>> get_metrics(name="SHD")
    [<class 'pgmpy.metrics.shd.SHD'>]
    """
    if isinstance(kwargs.get("name"), str):
        kwargs["name"] = kwargs["name"].lower()

    return all_objects(
        object_types=[BaseSupervisedMetric, BaseUnsupervisedMetric],
        package_name="pgmpy.metrics",
        return_names=False,
        filter_tags=kwargs,
    )
