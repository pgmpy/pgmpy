import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class SortnRegress(BaseCausalDiscovery):
    r"""
    Implementation of SortnRegress, a scale-invariant causal discovery method
    based on sorting variables by an ordering criterion and iteratively
    regressing each variable on its predecessors in that order. Two ordering
    criteria are supported via the ``criterion`` parameter:

    - ``criterion='r2'`` (default): orders variables by ascending global R²,
      based on the phenomenon that the explainable fraction of a variable's
      variance, captured by the coefficient of determination (R²), tends to
      increase along the causal order in linear additive noise models
      :cite:p:`Reisach2023`.
    - ``criterion='varsortability'``: orders variables by ascending marginal
      variance, based on the var-sortability phenomenon whereby variance
      tends to increase along the causal order :cite:p:`Reisach2021`.

    Only the ordering step (Step 2 below) differs between the two criteria;
    the edge-selection procedure (Steps 3-4) is identical for both.

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

    criterion : {'r2', 'varsortability'}, default='r2'
        The criterion used to compute the candidate causal ordering in Step 2.

        - ``'r2'``: order variables by ascending global R² (the original
          R²-SortnRegress algorithm).
        - ``'varsortability'``: order variables by ascending marginal
          variance, per the var-sortability phenomenon described in
          Reisach et al. (2021).

        The edge-selection procedure (Steps 3-4) is identical for both
        criteria; only the ordering step differs.

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
    - :cite:p:`Reisach2021`
    """

    def __init__(self, threshold=0.3, estimator=None, criterion="r2"):
        super().__init__()
        self.threshold = threshold
        self.estimator = estimator
        self.criterion = criterion

    def _fit(self, X):
        if self.criterion not in ("r2", "varsortability"):
            raise ValueError(f"criterion must be one of 'r2' or 'varsortability', got {self.criterion!r}.")
        if any(X.std() == 0):
            constant_cols = X.columns[X.std() == 0].tolist()
            raise ValueError(
                f"The following column(s) have zero variance (constant values): "
                f"{constant_cols}. Please drop these columns before fitting."
            )
        feature_names_in_ = list(X.columns)
        # clone the estimator or use default LinearRegression
        model_reg = clone(self.estimator) if self.estimator else LinearRegression()

        all_nodes_set = set(feature_names_in_)
        if self.criterion == "r2":
            order_values = {}
            for target in feature_names_in_:
                other_nodes = list(all_nodes_set - {target})

                y = X[target]
                predictors = X[other_nodes]

                model_reg.fit(predictors, y)
                predictions = model_reg.predict(predictors)

                residuals_variance = np.var(y - predictions)
                total_variance = np.var(y)

                order_values[target] = 1 - (residuals_variance / total_variance)
        else:  # "varsortability"
            order_values = X.var().to_dict()
        sorted_nodes = sorted(order_values, key=order_values.get)

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
            self.causal_graph_, nodelist=feature_names_in_, weight=1, dtype="int"
        )

        return self
