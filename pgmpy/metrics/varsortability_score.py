from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import SortnRegress
from pgmpy.metrics._base import BaseSupervisedMetric, BaseUnsupervisedMetric, get_metrics


class VarSortability(BaseUnsupervisedMetric):
    r"""
    Metric to compare the learned graph with a baseline variance sorting algorithm.

    Var-sortability measures how well marginal variances reflect the causal structure encoded in `causal_graph`. For
    each directed path in the graph, this metric checks whether variance increases monotonically along the path (or
    remains approximately equal). A score of 1.0 indicates perfect alignment: variances are non-decreasing along all
    causal paths.

    This metric is agnostic to how `causal_graph` was produced, so it can be used to evaluate the output of any causal
    discovery algorithm, or a ground-truth graph, against a given dataset.

    Parameters
    ----------
    metric: 'shd'
    variant: "r2"
    threshold: 0.3
    estimator: None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from pgmpy.metrics import VarSortability
    >>> np.random.seed(42)
    >>> n = 500
    >>> x = np.random.normal(0, 1.0, n)
    >>> y = 2.0 * x + np.random.normal(0, 0.5, n)
    >>> z = 2.0 * y + np.random.normal(0, 0.5, n)
    >>> data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    >>> dag = DAG([('X', 'Y'), ('Y', 'Z')])
    >>> metric = VarSortability()
    >>> score = metric.evaluate(X=data, causal_graph=dag)
    >>> score > 0.7
    True

    References
    ----------
    - :cite:p:`Reisach2021`
    """

    _tags = {
        "name": "sortability",
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
        # Step 1: Use the sortnregress algorithm to estimate a causal graph.
        est = SortnRegress(variant=self.variant, threshold=self.threshold, estimator=self.estimator)
        varsort_graph = est.fit(X).causal_graph_

        # Step 2: Use an supervised metric to compare the `causal_graph` with the one learned using sortnregress.
        metric_class = get_metrics(self.metric)
        if not isinstance(metric_class, BaseSupervisedMetric):
            raise ValueError("Incorrect metric")
        else:
            metric_est = metric_class().evaluate(causal_graph, varsort_graph)

        # Step 3: Return this metric.
        return metric_est
