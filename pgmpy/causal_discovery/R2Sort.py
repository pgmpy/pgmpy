import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pgmpy.causal_discovery._base import BaseOrderDiscovery
from pgmpy.utils import regression_residual_covariance


class R2Sort(BaseOrderDiscovery):
    r"""Causal discovery by sorting global R^2 values and regressing on predecessors.

    For each variable, fit a regression on all remaining variables and calculate
    its coefficient of determination, R^2. Sort variables by increasing R^2 to
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
        Regressor used to calculate global R^2 values and supply adaptive weights. It must implement ``fit`` and
        ``predict`` and expose ``coef_`` after fitting. If None, uses :class:`sklearn.linear_model.LinearRegression`.
        The estimator is cloned before fitting.

    return_type : str, default="dag"
        The graph type stored in ``causal_graph_``: ``"dag"`` or ``"pdag"``. The ``"pdag"`` option returns the completed
        PDAG representing the learned DAG's Markov equivalence class, so some edges can become undirected.

    Attributes
    ----------
    causal_order_ : list
        Estimated causal order obtained by sorting global R^2 values. This is the order used to construct the DAG before
        any conversion to a PDAG.

    causal_graph_ : pgmpy.base.DAG or pgmpy.base.PDAG
        The learned causal graph in the requested representation.

    adjacency_matrix_ : pandas.DataFrame
        Binary adjacency matrix in the input feature order. Directed edges have a one in the cause-to-effect entry;
        undirected edges have a one in both directions.

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

    _tags = {
        "name": "r2_sort",
        "data_types": ["continuous"],
        "identifiable_graph": "dag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": [],
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": False,
        "assumption:linearity": True,
        "assumption:additive_noise": True,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def _fit(self, X: pd.DataFrame) -> "R2Sort":
        return_type = self.return_type.lower()
        if return_type not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        model_reg = clone(self.estimator) if self.estimator is not None else LinearRegression()
        if self.estimator is None:
            correlation_matrix = np.atleast_2d(np.corrcoef(X.to_numpy(dtype=float), rowvar=False))
        r2_values = {}
        for column, target in enumerate(self.feature_names_in_):
            others = [other for other in range(len(self.feature_names_in_)) if other != column]
            if self.estimator is None:
                r2_values[target] = 1.0 - regression_residual_covariance(correlation_matrix, [column], others)[0, 0]
            else:
                model_reg.fit(X.iloc[:, others], X.iloc[:, column])
                r2_values[target] = r2_score(X.iloc[:, column], model_reg.predict(X.iloc[:, others]))

        causal_order = sorted(r2_values, key=r2_values.get)
        model = self._estimate_dag_from_causal_order(X, causal_order, regressor=model_reg)

        self.causal_order_ = list(causal_order)
        self.causal_graph_ = model if return_type == "dag" else model.to_pdag()
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.feature_names_in_)
        )
        return self
