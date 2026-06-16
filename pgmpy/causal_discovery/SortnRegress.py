import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class SortnRegress(_BaseCausalDiscovery):
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
    >>> data = pd.DataFrame(np.random.randn(1000, 3), columns=['X', 'Y', 'Z'])
    >>> data['Y'] += 2 * data['X']
    >>> data['Z'] += 3 * data['Y']
    >>> model = SortnRegress(threshold=0.3)
    >>> model.fit(data)
    >>> model.causal_graph_.edges()
    [('X', 'Y'), ('Y', 'Z')]

    References
    ----------
    .. [1] Reisach, A. G., Tami, M., Chambaz, A., Seiler, C., & Weichwald, S. (2023).
       A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models.
       Advances in Neural Information Processing Systems, 36.
       https://arxiv.org/abs/2303.18211
    """

    def __init__(self, threshold=0.3, estimator=None):
        super().__init__()
        self.threshold = threshold
        self.estimator = estimator

    def _fit(self, X):
        """
        The fitting procedure for the SortnRegress algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.
        Returns
        -------
        self : pgmpy.causal_discovery.SortnRegress
            Returns the instance with the fitted attributes.
        """
        feature_name_list = list(X.columns)
        # clone the estimator or use default LinearRegression
        model_reg = clone(self.estimator) if self.estimator else LinearRegression()

        r2_values = {}
        for target in feature_name_list:
            other_nodes = [node for node in feature_name_list if node != target]

            y = X[target].values.astype(float)
            predictors = X[other_nodes].values.astype(float)

            model_reg.fit(predictors, y)
            predictions = model_reg.predict(predictors)

            residuals_variance = np.var(y - predictions)
            total_variance = np.var(y)

            if total_variance == 0:
                r2_values[target] = 0.0
            else:
                r2_values[target] = 1 - (residuals_variance / total_variance)

        sorted_nodes = sorted(r2_values, key=r2_values.get)

        model = DAG()
        model.add_nodes_from(sorted_nodes)

        for i in range(1, len(sorted_nodes)):
            target = sorted_nodes[i]
            potential_parents = sorted_nodes[:i]

            y = X[target].values.astype(float)
            predictors = X[potential_parents].values.astype(float)

            model_reg.fit(predictors, y)
            coefs = model_reg.coef_

            for idx, coef in enumerate(coefs):
                if abs(coef) >= self.threshold:
                    model.add_edge(potential_parents[idx], target)

        self.causal_graph_ = model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, nodelist=feature_name_list, weight=1, dtype="int"
        )

        return self
