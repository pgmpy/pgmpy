import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class SortnRegress(BaseCausalDiscovery):
    r"""
    Implementation of the R²-SortnRegress algorithm for causal discovery.

    R²-SortnRegress is a scale-invariant causal discovery method based on the
    phenomenon that the explainable fraction of a variable's variance, captured
    by the coefficient of determination (R²), tends to increase along the
    causal order in linear additive noise.

    Given an :math:`n \times d` dataset :math:`\mathbf{X}` with columns
    :math:`X_1, \dots, X_d`, the algorithm proceeds as follows:

    1. **Global R² Estimation**: For each variable :math:`X_t`, fit a linear
       regression using all remaining variables :math:`\mathbf{X}_{\setminus \{t\}}`
       as predictors to calculate its global R² value:

       .. math::

           R^2(X_t) = 1 - \frac{\text{Var}(X_t - \widehat{X}_t)}{\text{Var}(X_t)}

    2. **Candidate Causal Ordering**: Sort the variables in ascending order
       of their estimated global R² values to form a candidate topological
       ordering :math:`\pi`:

       .. math::

           R^2(X_{\pi(1)}) \leq R^2(X_{\pi(2)}) \leq \dots \leq R^2(X_{\pi(d)})

    3. Iterative Regression: For each target node :math:`X_{\pi(i)}`
       (for :math:`i = 2, \dots, d`), fit a linear regression on all preceding
       variables (potential parents :math:`X_{\pi(1)}, \dots, X_{\pi(i-1)}`):

       .. math::

           X_{\pi(i)} = \sum_{j=1}^{i-1} \beta_{j,\pi(i)} X_{\pi(j)}
                        + \varepsilon_{\pi(i)}

       where :math:`\varepsilon_{\pi(i)}` is the noise term and
       :math:`\beta_{j,\pi(i)}` are the regression coefficients.

    4. Edge Selection: Add a directed edge :math:`X_{\pi(j)} \to X_{\pi(i)}`
       if:

       .. math::

           |\beta_{j,\pi(i)}| \geq \texttt{threshold}

    Parameters
    ----------
    threshold : float, default=0.3
        The absolute value threshold for regression coefficients. Edges with
        coefficients below this value are pruned to sparsify the graph.
        A default of 0.3 is chosen to align with the benchmarking settings
        described in Reisach et al. (2023).

    estimator : sklearn-style regression estimator, default=None
        The regression estimator instance to use for edge selection.
        If None, defaults to sklearn.linear_model.LinearRegression().

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_: int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_: np.ndarray
        The feature names in the dataset used to learn the causal graph.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.causal_discovery import SortnRegress
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(np.random.randn(1000, 3), columns=['X', 'Y', 'Z'])
    >>> data['Z'] += 2.5 * data['X'] + 2.5 * data['Y']
    >>> model = SortnRegress(threshold=0.3)
    >>> _ = model.fit(data)
    >>> list(model.causal_graph_.edges())
    [('X', 'Z'), ('Y', 'Z')]

    References
    ----------
    - :cite:p:`Reisach2023`
    """

    def __init__(self, threshold=0.3, estimator=None):
        super().__init__()
        self.threshold = threshold
        self.estimator = estimator

    def _fit(self, X):
        if any(X.std() == 0):
            constant_cols = X.columns[X.std() == 0].tolist()
            raise ValueError(
                f"The following column(s) have zero variance (constant values): "
                f"{constant_cols}. Please drop these columns before fitting."
            )
        self.feature_names_in_ = list(X.columns)
        # clone the estimator or use default LinearRegression
        model_reg = clone(self.estimator) if self.estimator else LinearRegression()

        all_nodes_set = set(self.feature_names_in_)

        r2_values = {}
        for target in self.feature_names_in_:
            other_nodes = list(all_nodes_set - {target})

            y = X[target]
            predictors = X[other_nodes]

            model_reg.fit(predictors, y)
            predictions = model_reg.predict(predictors)

            residuals_variance = np.var(y - predictions)
            total_variance = np.var(y)

            r2_values[target] = 1 - (residuals_variance / total_variance)

        sorted_nodes = sorted(r2_values, key=r2_values.get)

        model = DAG()
        model.add_nodes_from(sorted_nodes)

        for i in range(1, len(sorted_nodes)):
            target = sorted_nodes[i]
            potential_parents = sorted_nodes[:i]

            y = X[target]
            predictors = X[potential_parents]

            model_reg.fit(predictors, y)
            coefs = model_reg.coef_

            for idx, coef in enumerate(coefs):
                if abs(coef) >= self.threshold:
                    model.add_edge(potential_parents[idx], target)

        self.causal_graph_ = model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, nodelist=self.feature_names_in_, weight=1, dtype="int"
        )

        return self

    def varsortability(self, X, tol=1e-9):
        r"""
        Compute the var-sortability of the input data relative to the graph
        discovered by this estimator.

        Var-sortability measures how well marginal variances reflect the causal
        structure. For each directed path in the true DAG, this metric checks whether
        variance increases monotonically along the path (or remains approximately equal).
        A score of 1.0 indicates perfect alignment: variances are non-decreasing along
        all causal paths.

        The metric is based on the observation that under linear additive noise
        models, the variance of a variable is influenced by the variances of its
        ancestors and the noise variance. If causal structure holds, we expect
        variance to accumulate downstream in the causal graph.

        Parameters
        ----------
        X : pd.DataFrame
            The observed data matrix (same data used for fitting).

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
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.causal_discovery import SortnRegress
        >>> np.random.seed(42)
        >>> n = 500
        >>> x = np.random.normal(0, 1.0, n)
        >>> y = 2.0 * x + np.random.normal(0, 0.5, n)
        >>> z = 2.0 * y + np.random.normal(0, 0.5, n)
        >>> data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})

        >>> sr = SortnRegress()
        >>> sr.fit(data)
        >>> result = sr.varsortability(data)
        >>> 'varsortability' in result
        True
        >>> result['varsortability'] > 0.7
        True

        References
        ----------
        - :cite:p:`Reisach2023`
        """
        if not hasattr(self, "causal_graph_"):
            raise ValueError("Call .fit(X) before computing varsortability.")

        nodes = self.feature_names_in_
        x_mat = X[nodes].values

        d = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        W = np.zeros((d, d))
        for u, v in self.causal_graph_.edges():
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
            return {"varsortability": 1.0}  # Returns dict

        return {"varsortability": float(n_ordered_path / n_paths)}  # Returns di
