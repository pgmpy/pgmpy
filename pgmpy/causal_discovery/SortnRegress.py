import networkx as nx
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class SortnRegress(_BaseCausalDiscovery):
    """
    Implementation of the SortnRegress algorithm for causal discovery.

    SortnRegress is based on the phenomenon of "varsortability," where in many
    linear additive noise models, the causal order of variables is correlated
    with the order of their marginal variances.

    Given an n x d dataset X with columns X_1, ..., X_d, the algorithm
    proceeds as follows:

    1. Variance Ordering: Compute the marginal variance of each variable and
       sort them in ascending order to obtain a permutation π such that:

           Var(X_π(1)) ≤ Var(X_π(2)) ≤ ... ≤ Var(X_π(d))

    2. Iterative Regression: For each target node X_π(i) (i = 2, ..., d),
       fit a linear regression on all preceding variables (potential parents
       X_π(1), ..., X_π(i-1)):

           X_π(i) = β_1·X_π(1) + β_2·X_π(2) + ... + β_{i-1}·X_π(i-1) + ε_π(i)

       where ε_π(i) is the noise term and β_j are the regression coefficients.

    3. Edge Selection: Add a directed edge X_π(j) → X_π(i) if:

           |β_j| ≥ threshold

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
    Reisach, A. G., Seiler, C., & Weichwald, S. (2021). Beware of the Simulated DAG!
    Causal Discovery Benchmarks May Be Easy To Game. Advances in Neural Information
    Processing Systems, 34. https://arxiv.org/abs/2102.13647
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
