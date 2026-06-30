import numpy as np
import pandas as pd

from pgmpy.base import DAG, PDAG
from pgmpy.metrics._base import BaseUnsupervisedMetric


class VarSortability(BaseUnsupervisedMetric):
    r"""
    Compute the var-sortability score of a dataset relative to a causal DAG.

    Var-sortability measures how well marginal variances reflect the causal
    structure. For each directed path in the true DAG, this metric checks whether
    variance increases monotonically along the path (or remains approximately equal).
    A score of 1.0 indicates perfect alignment: variances are non-decreasing along
    all causal paths.

    The metric is based on the observation that under linear additive noise models,
    the variance of a variable is influenced by the variances of its ancestors and
    the noise variance. If causal structure holds, we expect variance to accumulate
    downstream in the causal graph.

    Parameters
    ----------
    tol : float, default=1e-9
        Tolerance for checking near-equality of variances. When comparing
        ``Var(target) / Var(source)``, values in the range ``[1-tol, 1+tol]``
        are treated as "approximately equal" and weighted as 0.5.

    Returns
    -------
    Dict[str, float]
        Dictionary with key ``'varsortability'`` containing the score in [0, 1].

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.base import DAG
    >>> from pgmpy.metrics import VarSortability

    >>> # Data where variance grows along causal paths
    >>> np.random.seed(42)
    >>> n = 1000
    >>> x = np.random.normal(0, 1, n)
    >>> y = x + np.random.normal(0, 0.5, n)  # Child has higher variance
    >>> z = y + np.random.normal(0, 0.5, n)
    >>> X = pd.DataFrame({'X': x, 'Y': y, 'Z': z})

    >>> true_dag = DAG([('X', 'Y'), ('Y', 'Z')])
    >>> scorer = VarSortability()
    >>> result = scorer.evaluate(X, true_dag)
    >>> result['varsortability']  # Should be high (> 0.8)

    Notes
    -----
    Var-sortability is particularly useful for evaluating causal discovery methods
    in linear models. High varsortability suggests that the discovered causal
    structure aligns with variance properties in the data.

    References
    ----------
    .. [1] Reisach, A. G., Seiler, C., & Weichwald, S. (2021).
       Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game.
       Advances in Neural Information Processing Systems, 34.
    """

    _tags = {
        "name": "VarSortability",
        "requires_true_graph": True,
        "requires_data": True,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_gragh_types": (DAG, PDAG),
    }

    def __init__(self, tol=1e-9):
        self.tol = tol
        super().__init__()

    def _evaluate(self, X, causal_graph):
        """
        Evaluate varsortability by checking variance ordering along causal paths.

        Parameters
        ----------
        X : pd.DataFrame
            The observed data matrix.

        causal_graph : pgmpy.base.DAG or pgmpy.base.PDAG
            The (true) causal graph to evaluate against.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'varsortability' key containing the score in [0, 1].
        """
        # extract nodes and adjacency matrix
        if isinstance(X, pd.DataFrame):
            nodes = list(X.columns)
            x_mat = X.values
        else:
            nodes = list(range(X.shape[1]))
            x_mat = X

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

            n_ordered_path += (variance_ratio > 1 + self.tol).sum()

            n_ordered_path += 0.5 * ((variance_ratio <= 1 + self.tol) * (variance_ratio > 1 - self.tol)).sum()

            Ek = Ek.dot(E)

        if n_paths == 0:
            return 1.0

        return n_ordered_path / n_paths
