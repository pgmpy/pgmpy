import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class SortnRegress(_BaseCausalDiscovery):
    """
    Implementation of the SortnRegress algorithm for causal discovery.

    SortnRegress is based on the phenomenon of "varsortability," where in many
    linear additive noise models, the causal order of variables is correlated
    with the order of their marginal variances.

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
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)
        self.variables_ = list(X.columns)

        model_reg = self.estimator if self.estimator else LinearRegression()

        variances = X.var().sort_values()
        sorted_nodes = variances.index.tolist()

        model = DAG()
        model.add_nodes_from(sorted_nodes)

        for i in range(1, len(sorted_nodes)):
            target = sorted_nodes[i]
            potential_parents = sorted_nodes[:i]

            y = X[target].values
            predictors = X[potential_parents].values

            model_reg.fit(predictors, y)
            coefs = model_reg.coef_

            for idx, coef in enumerate(coefs):
                if abs(coef) >= self.threshold:
                    model.add_edge(potential_parents[idx], target)

        self.causal_graph_ = model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, nodelist=self.variables_, weight=1, dtype="int"
        )

        return self
