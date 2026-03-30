import networkx as nx
import numpy as np

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
        described in Reisach et al. (2021), where it serves as a robust
        cutoff for identifying significant causal links in sparse models.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    References
    ----------
    Reisach, A. G., Seiler, C., & Weichwald, S. (2021). Beware of the Simulated DAG!
    Causal Discovery Benchmarks May Be Easy To Game. Advances in Neural Information
    Processing Systems, 34. https://arxiv.org/abs/2102.13647
    """

    def __init__(self, threshold=0.3):
        super().__init__()
        self.threshold = threshold

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
        self.variables = list(X.columns)

        variances = X.var().sort_values()
        sorted_nodes = variances.index.tolist()

        model = DAG()
        model.add_nodes_from(sorted_nodes)

        for i in range(1, len(sorted_nodes)):
            target = sorted_nodes[i]
            potential_parents = sorted_nodes[:i]

            y = X[target].values.astype(float)
            predictors = X[potential_parents].values.astype(float)

            coefs, _, _, _ = np.linalg.lstsq(predictors, y, rcond=None)

            for idx, coef in enumerate(coefs):
                if abs(coef) >= self.threshold:
                    model.add_edge(potential_parents[idx], target)

        self.causal_graph_ = model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, nodelist=self.variables, weight=1, dtype="int"
        )

        return self
