import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects


class _BaseSupervisedMetric(BaseObject):
    """
    Base class for all metric classes in pgmpy that require ground truth causal graph.
    """

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
        if not isinstance(true_causal_graph, self._tags["supported_graph_types"]):
            raise ValueError(
                f"The true_causal_graph must be one of the following types: "
                f"{self._tags['supported_graph_types']}, "
                f"but got {type(true_causal_graph)} instead."
            )

        if not isinstance(est_causal_graph, self._tags["supported_graph_types"]):
            raise ValueError(
                f"The est_causal_graph must be one of the following types: "
                f"{self._tags['supported_graph_types']}, "
                f"but got {type(est_causal_graph)} instead."
            )

        if not set(true_causal_graph.nodes()) == set(est_causal_graph.nodes()):
            raise ValueError(
                "The `true_causal_graph` and `est_causal_graph` must be on the same nodes."
            )

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


class _BaseUnsupervisedMetric(BaseObject):
    """
    Base class for all metric classes in pgmpy that do not require ground truth causal graph.
    """

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
        if not isinstance(causal_graph, self._tags["supported_graph_types"]):
            raise ValueError(
                f"The causal_graph must be one of the following types: "
                f"{self._tags['supported_graph_types']}, "
                f"but got {type(causal_graph)} instead."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"The data must be a pandas.DataFrame instance, "
                f"but got {type(X)} instead."
            )
        elif len(set(causal_graph.nodes()) - set(X.columns)) > 0:
            raise ValueError(
                "Missing columns in data. Can't find values for the following variables: "
                f" {set(causal_graph.nodes()) - set(X.columns)}"
            )

        return self._evaluate(X=X, causal_graph=causal_graph, **kwargs)

    def __call__(self, X, causal_graph, **kwargs):
        return self.evaluate(X=X, causal_graph=causal_graph, **kwargs)


def get_metrics(**kwargs):
    """
    Get metric classes matching the given tag filters.

    Parameters
    ----------
    **kwargs
        Keyword arguments specifying tag filters to be passed to
        :func:`skbase.lookup.all_objects` via its ``filter_tags`` parameter.

    Returns
    -------
    Type[BaseObject] or list[Type[BaseObject]]
        Metric class(es) corresponding to the given tag filters.

    Raises
    ------
    ValueError
        If no metric class matching the given tag filters is found.
    """
    return all_objects(
        object_types=[_BaseSupervisedMetric, _BaseUnsupervisedMetric],
        package_name="pgmpy.metrics",
        return_names=False,
        filter_tags=kwargs,
    )
