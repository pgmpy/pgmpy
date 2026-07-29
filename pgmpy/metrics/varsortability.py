from pgmpy.base import DAG, PDAG
from pgmpy.metrics._base import BaseSupervisedMetric, BaseUnsupervisedMetric, get_metrics


class VarSortability(BaseUnsupervisedMetric):
    r"""
    Metric to compare the learned graph with a baseline variance sorting algorithm.

    This metric quantifies how close a given `causal_graph` is to the graph that
    :class:`~pgmpy.causal_discovery.SortnRegress` recovers from the same dataset. It fits `SortnRegress` on
    `X` and then compares the resulting graph against `causal_graph` using a supervised graph-distance metric
    (Structural Hamming Distance by default).

    Parameters
    ----------
    metric : str, default='shd'
        Name of the supervised metric used to compare `causal_graph` against the graph learned by
        `SortnRegress`.
    variant : {'r2', 'varsortability'}, default='r2'
        Ordering criterion passed through to `SortnRegress`. See
        :class:`~pgmpy.causal_discovery.SortnRegress` for details.
    threshold: float, default=0.3
        Threshold passed through to `SortnRegress`. See :class:`~pgmpy.causal_discovery.SortnRegress` for details.
    estimator: sklearn-style regression estimator, default=None
        Regression estimator passed through to `SortnRegress`. If None, `SortnRegress` defaults to
        sklearn.linear_model.LinearRegression().

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from pgmpy.metrics import VarSortability
    >>> rng = np.random.default_rng(seed=42)
    >>> n = 500
    >>> x = rng.normal(0, 1.0, n)
    >>> y = 2.0 * x + rng.normal(0, 0.5, n)
    >>> z = 2.0 * y + rng.normal(0, 0.5, n)
    >>> data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    >>> dag = DAG([('X', 'Y'), ('Y', 'Z')])
    >>> metric = VarSortability()
    >>> score = metric.evaluate(X=data, causal_graph=dag)
    0

    References
    ----------
    - :cite:p:`Reisach2021`
    """

    _tags = {
        "name": "varsortability",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,
        "supported_graph_types": (DAG, PDAG),
        "is_default": False,
    }

    def __init__(self, metric="shd", variant="r2", threshold=0.3, estimator=None):
        self.metric = metric
        self.variant = variant
        self.threshold = threshold
        self.estimator = estimator
        super().__init__()

    def _evaluate(self, X, causal_graph, **kwargs):
        from pgmpy.causal_discovery import SortnRegress

        # Step 1: Use the sortnregress algorithm to estimate a causal graph.
        est = SortnRegress(variant=self.variant, threshold=self.threshold, estimator=self.estimator)
        varsort_graph = est.fit(X).causal_graph_

        # Step 2: Use an supervised metric to compare the `causal_graph` with the one learned using sortnregress.
        metric_class = get_metrics(name=self.metric)

        if not isinstance(metric_class(), BaseSupervisedMetric):
            raise ValueError(f"Metric '{self.metric}' is not a supported supervised metric.")
        else:
            metric_est = metric_class().evaluate(causal_graph, varsort_graph)

        # Step 3: Return this metric.
        return metric_est
