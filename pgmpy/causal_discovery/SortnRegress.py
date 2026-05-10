import networkx as nx
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class SortnRegress(_BaseCausalDiscovery):
    r"""
    Implementation of the SortnRegress algorithm for causal discovery.

    SortnRegress is based on the phenomenon of "varsortability," where in many
    linear additive noise models, the causal order of variables is correlated
    with the order of their marginal variances.

    Given an :math:`n \times d` dataset :math:`\mathbf{X}` with columns
    :math:`X_1, \dots, X_d`, the algorithm proceeds as follows:

    1. Variance Ordering: Compute the marginal variance of each variable and
       sort them in ascending order to obtain a permutation :math:`\pi` such that:

       .. math::

           \widehat{\text{Var}}(X_{\pi(1)}) \leq
           \widehat{\text{Var}}(X_{\pi(2)}) \leq \dots \leq
           \widehat{\text{Var}}(X_{\pi(d)})

    2. Iterative Regression: For each target node :math:`X_{\pi(i)}`
       (for :math:`i = 2, \dots, d`), fit a linear regression on all preceding
       variables (potential parents :math:`X_{\pi(1)}, \dots, X_{\pi(i-1)}`):

       .. math::

           X_{\pi(i)} = \sum_{j=1}^{i-1} \beta_{j,\pi(i)} X_{\pi(j)}
                        + \varepsilon_{\pi(i)}

       where :math:`\varepsilon_{\pi(i)}` is the noise term and
       :math:`\beta_{j,\pi(i)}` are the regression coefficients.

    3. Edge Selection: Add a directed edge :math:`X_{\pi(j)} \to X_{\pi(i)}`
       if:

       .. math::

           |\beta_{j,\pi(i)}| \geq \texttt{threshold}

    Parameters
    ----------
    threshold : float, default=0.3
        The absolute value threshold for regression coefficients. Edges with
        coefficients below this value are pruned to sparsify the graph.
        A default of 0.3 is chosen to align with the benchmarking settings
        described in Reisach et al. (2021).

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
    .. [1] Reisach, A. G., Seiler, C., & Weichwald, S. (2021). Beware of the
       Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game.
       Advances in Neural Information Processing Systems, 34.
       https://arxiv.org/abs/2102.13647
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

        variances = X.var().sort_values()
        sorted_nodes = variances.index.tolist()

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
