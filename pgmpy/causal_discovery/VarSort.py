import pandas as pd

from pgmpy.causal_discovery._base import BaseOrderDiscovery


class VarSort(BaseOrderDiscovery):
    r"""Causal discovery by sorting marginal variances and regressing on predecessors.

    Sort variables by increasing marginal variance to obtain an estimated
    causal order, exploiting the varsortability pattern studied in :cite:p:`Reisach2021`.
    Equal variances retain their input column order. Standardizing or rescaling
    the variables can change this causal order and the recovered graph.

    The shared graph-estimation step regresses each variable on its predecessors.
    Absolute regression coefficients supply adaptive-Lasso weights, and BIC
    selects the Lasso penalty. Nonzero coefficients determine the DAG's edges.

    Parameters
    ----------
    estimator : sklearn-style regression estimator, default=None
        Regressor supplying adaptive weights through its ``coef_`` attribute.
        If None, uses :class:`sklearn.linear_model.LinearRegression`. The estimator
        is cloned before fitting; it does not affect the estimated causal order.

    return_type : str, default="dag"
        The graph type stored in ``causal_graph_``: ``"dag"`` or ``"pdag"``.
        The ``"pdag"`` option returns the completed PDAG representing the learned
        DAG's Markov equivalence class, so some edges can become undirected.

    Attributes
    ----------
    causal_order_ : list
        Estimated causal order obtained by sorting marginal variances.
        This is the order used to construct the DAG before any conversion to a PDAG.

    causal_graph_ : pgmpy.base.DAG or pgmpy.base.PDAG
        The learned causal graph in the requested representation.

    adjacency_matrix_ : pandas.DataFrame
        Binary adjacency matrix in the input feature order. Directed edges have
        a one in the cause-to-effect entry; undirected edges have a one in both
        directions.

    n_features_in_ : int
        Number of features in the data used to learn the graph.

    feature_names_in_ : numpy.ndarray
        Names of the features in the data used to learn the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import VarSort
    >>> rng = np.random.default_rng(42)
    >>> data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["X", "Y", "Z"])
    >>> data["Z"] += 2.5 * data["X"] + 2.5 * data["Y"]
    >>> model = VarSort().fit(data)
    >>> sorted(model.causal_graph_.edges())
    [('X', 'Z'), ('Y', 'Z')]

    See Also
    --------
    R2Sort : Causal discovery using global R² sorting.

    References
    ----------
    - :footcite:t:`Reisach2021`
    """

    def _fit(self, X: pd.DataFrame) -> "VarSort":
        """Estimate a causal order from marginal variances, then learn the graph."""
        order_values = X.var().to_dict()
        causal_order = sorted(order_values, key=order_values.get)
        return self._fit_from_causal_order(X, causal_order)
