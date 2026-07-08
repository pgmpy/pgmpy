import numpy as np

from pgmpy.base import DAG
from pgmpy.metrics._base import BaseUnsupervisedMetric


def compute_varsortability(X, causal_graph, tol=1e-9):
    r"""
    Compute the var-sortability of a dataset relative to a given causal graph.

    Var-sortability measures how well marginal variances reflect the causal
    structure. For each directed path in the graph, this metric checks whether
    variance increases monotonically along the path (or remains approximately
    equal). A score of 1.0 indicates perfect alignment: variances are
    non-decreasing along all causal paths.

    This is a diagnostic of the *relationship between a dataset and a graph*,
    not an algorithm for producing a graph, and it is agnostic to how the
    graph was produced. It is commonly used to sanity-check benchmark
    datasets: a high var-sortability score on the ground-truth DAG suggests
    the dataset may be too "easy" for sorting-based causal discovery methods,
    independent of which algorithm is used to recover it.

    Parameters
    ----------
    X : pd.DataFrame
        The observed data matrix. Columns must include every node in
        `causal_graph`.

    causal_graph : Instance of type pgmpy.base
        The causal graph to evaluate the data against. This can be a
        ground-truth DAG, or the output of any causal discovery algorithm.

    tol : float, default=1e-9
        Tolerance for checking near-equality of variances. When comparing
        ``Var(target) / Var(source)``, values in the range ``[1-tol, 1+tol]``
        are treated as "approximately equal" and weighted as 0.5.

    Returns
    -------
    float
        The var-sortability score, in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from pgmpy.metrics.varsortability_score import compute_varsortability
    >>> np.random.seed(42)
    >>> n = 500
    >>> x = np.random.normal(0, 1.0, n)
    >>> y = 2.0 * x + np.random.normal(0, 0.5, n)
    >>> z = 2.0 * y + np.random.normal(0, 0.5, n)
    >>> data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    >>> dag = DAG([('X', 'Y'), ('Y', 'Z')])
    >>> score = compute_varsortability(data, dag)
    >>> score > 0.7
    True

    References
    ----------
    - :cite:p:`Reisach2021`
    """
    nodes = list(causal_graph.nodes())
    x_mat = X[nodes].values

    d = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    W = np.zeros((d, d))
    for u, v in causal_graph.edges():
        if u in node_to_idx and v in node_to_idx:
            W[node_to_idx[u], node_to_idx[v]] = 1
    E = W != 0

    # compute marginal variances
    var = np.var(x_mat, axis=0, keepdims=True)

    n_paths = 0
    n_ordered_path = 0
    Ek = E.copy()

    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        variance_ratio = Ek * (var / var.T)

        n_ordered_path += (variance_ratio > 1 + tol).sum()

        n_ordered_path += 0.5 * ((variance_ratio <= 1 + tol) * (variance_ratio > 1 - tol)).sum()

        Ek = Ek.dot(E)

    if n_paths == 0:
        return 1.0

    return float(n_ordered_path / n_paths)


class VarSortability(BaseUnsupervisedMetric):
    r"""
    Var-sortability metric for evaluating a causal graph against observed data.

    Var-sortability measures how well marginal variances reflect the causal
    structure encoded in `causal_graph`. For each directed path in the graph,
    this metric checks whether variance increases monotonically along the
    path (or remains approximately equal). A score of 1.0 indicates perfect
    alignment: variances are non-decreasing along all causal paths.

    This metric is agnostic to how `causal_graph` was produced, so it can be
    used to evaluate the output of any causal discovery algorithm, or a
    ground-truth graph, against a given dataset.

    Parameters
    ----------
    tol : float, default=1e-9
        Tolerance for checking near-equality of variances. When comparing
        ``Var(target) / Var(source)``, values in the range ``[1-tol, 1+tol]``
        are treated as "approximately equal" and weighted as 0.5.

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

    _tags = {"supported_graph_types": (DAG,)}

    def __init__(self, tol=1e-9):
        self.tol = tol
        super().__init__()

    def _evaluate(self, X, causal_graph, **kwargs):
        return compute_varsortability(X=X, causal_graph=causal_graph, tol=self.tol)
