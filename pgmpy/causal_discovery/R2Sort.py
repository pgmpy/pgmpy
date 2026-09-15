import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pgmpy.causal_discovery._base import BaseOrderDiscovery


class R2Sort(BaseOrderDiscovery):
    r"""Causal discovery by sorting global R² values and regressing on predecessors.

    For each variable, fit a regression on all remaining variables and calculate
    its coefficient of determination, R². Sort variables by increasing R² to
    obtain an estimated causal order :cite:p:`Reisach2023`. Equal scores retain
    their input column order.

    The shared graph-estimation step regresses each variable on its predecessors.
    Absolute regression coefficients supply adaptive-Lasso weights, and BIC
    selects the Lasso penalty. Nonzero coefficients determine the DAG's edges.
    With the default linear regressor, the ordering and edge selection are
    invariant to rescaling individual variables, apart from numerical effects.

    Parameters
    ----------
    estimator : sklearn-style regression estimator, default=None
        Regressor used to calculate global R² values and supply adaptive weights.
        It must implement ``fit`` and ``predict`` and expose ``coef_`` after fitting.
        If None, uses :class:`sklearn.linear_model.LinearRegression`. The estimator
        is cloned before fitting.

    return_type : str, default="dag"
        The graph type stored in ``causal_graph_``: ``"dag"`` or ``"pdag"``.
        The ``"pdag"`` option returns the completed PDAG representing the learned
        DAG's Markov equivalence class, so some edges can become undirected.

    Attributes
    ----------
    causal_order_ : list
        Estimated causal order obtained by sorting global R² values.
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
    >>> from pgmpy.causal_discovery import R2Sort
    >>> rng = np.random.default_rng(42)
    >>> data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["X", "Y", "Z"])
    >>> data["Z"] += 2.5 * data["X"] + 2.5 * data["Y"]
    >>> model = R2Sort().fit(data)
    >>> sorted(model.causal_graph_.edges())
    [('X', 'Z'), ('Y', 'Z')]

    See Also
    --------
    VarSort : Causal discovery using marginal-variance sorting.

    References
    ----------
    - :footcite:t:`Reisach2023`
    - :footcite:t:`Reisach2021`
    """

    def _fit(self, X: pd.DataFrame) -> "R2Sort":
        """Estimate a causal order from global R² values, then learn the graph."""
        model_reg = clone(self.estimator) if self.estimator is not None else LinearRegression()
        all_nodes_set = set(self.feature_names_in_)
        order_values = {}
        for target in self.feature_names_in_:
            other_nodes = list(all_nodes_set - {target})
            y = X[target]
            predictors = X[other_nodes]

            model_reg.fit(predictors, y)
            predictions = model_reg.predict(predictors)
            order_values[target] = r2_score(y, predictions)

        causal_order = sorted(order_values, key=order_values.get)
        return self._fit_from_causal_order(X, causal_order, regressor=model_reg)
