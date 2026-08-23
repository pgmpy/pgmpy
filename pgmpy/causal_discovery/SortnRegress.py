import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.metrics import r2_score

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class SortnRegress(BaseCausalDiscovery):
    r"""
    Implementation of SortnRegress, a causal discovery method based on sorting variables by an ordering
    criterion and iteratively regressing each variable on its predecessors in that order. Two ordering
    criteria are supported via the ``variant`` parameter:

    - ``variant='r2'`` (default): orders variables by ascending global R², based on the phenomenon that the
      explainable fraction of a variable's variance, captured by the coefficient of determination (R²), tends to
      increase along the causal order in linear additive noise models :cite:p:`Reisach2023`. R² is invariant to
      rescaling of the columns of :math:`\mathbf{X}`, so this variant is scale-invariant end to end.

    - ``variant='varsortability'``: orders variables by ascending marginal variance, based on the var-sortability
      phenomenon whereby variance tends to increase along the causal order :cite:p:`Reisach2021`. Marginal
      variance depends on the measurement scale, so this variant is **not** scale-invariant: standardizing or
      otherwise rescaling the data may change the recovered graph.

    Step 1 differs between the two criteria; the remaining steps are identical for both. For
    ``variant='varsortability'``, Step 1 is replaced by computing the marginal variance
    :math:`\text{Var}(X_t)` of each variable.

    Given an :math:`n \times d` dataset :math:`\mathbf{X}` with columns :math:`X_1, \dots, X_d`, the algorithm proceeds
    as follows:

    1. **Global R² Estimation**: For each variable :math:`X_t`, fit a linear regression using all remaining variables
       :math:`\mathbf{X}_{\setminus t}` as predictors to calculate its global R² value:

       .. math::

           R^2(X_t) = 1 - \frac{\text{Var}(X_t - \widehat{X}_t)}{\text{Var}(X_t)}

    2. **Candidate Causal Ordering**: Sort the variables in ascending order of their estimated global R² values to form
       a candidate topological ordering :math:`\pi`:

       .. math::

           R^2(X_{\pi(1)}) \leq R^2(X_{\pi(2)}) \leq \dots \leq R^2(X_{\pi(d)})

    3. **Iterative Regression**: For each target node :math:`X_{\pi(i)}` (for :math:`i = 2, \dots, d`), fit a linear
       regression on all preceding variables (potential parents :math:`X_{\pi(1)}, \dots, X_{\pi(i-1)}`):

       .. math::

           X_{\pi(i)} = \sum_{j=1}^{i-1} \beta_{j,\pi(i)} X_{\pi(j)} + \varepsilon_{\pi(i)}

       where :math:`\varepsilon_{\pi(i)}` is the noise term and :math:`\beta_{j,\pi(i)}` are the regression
       coefficients.

    4. **Edge Selection**: Prune edges using an adaptive Lasso :cite:p:`Reisach2021`. The absolute
       least-squares coefficients :math:`|\beta_{j,\pi(i)}|` serve as adaptive weights, and an L1
       penalty with the regularization parameter selected by the Bayesian Information Criterion is
       fit on the reweighted predictors. An edge :math:`X_{\pi(j)} \to X_{\pi(i)}` is added if the
       resulting coefficient is non-zero. This step is invariant to rescaling of the predictor columns:
       a least-squares coefficient scales inversely with its predictor, so under
       :math:`X_{\pi(j)} \mapsto c_j X_{\pi(j)}` the weight becomes
       :math:`|\beta_{j,\pi(i)}| / c_j` and the reweighted predictor
       :math:`c_j X_{\pi(j)} \cdot |\beta_{j,\pi(i)}| / c_j` is unchanged.

    Parameters
    ----------

    estimator : sklearn-style regression estimator, default=None
        The least-squares estimator used to compute the global R² values and to supply the
        adaptive weights. If None, defaults to sklearn.linear_model.LinearRegression().

    variant : {'r2', 'varsortability'}, default='r2'
        The variant used to compute the candidate causal ordering in Step 2.

        - ``'r2'``: order variables by ascending global R² (the original R²-SortnRegress algorithm).
        - ``'varsortability'``: order variables by ascending marginal variance, per the var-sortability phenomenon
          described in :cite:p:`Reisach2021`

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
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=['X', 'Y', 'Z'])
    >>> data['Z'] += 2.5 * data['X'] + 2.5 * data['Y']
    >>> model = SortnRegress()
    >>> _ = model.fit(data)
    >>> list(model.causal_graph_.edges())
    [('X', 'Z'), ('Y', 'Z')]

    References
    ----------
    - :footcite:t:`Reisach2023`
    - :footcite:t:`Reisach2021`
    """

    def __init__(self, variant="r2", estimator=None):
        super().__init__()
        self.estimator = estimator
        self.variant = variant

    def _fit(self, X):
        # Step 0: Validate the input arguments and initialize regressor.
        if self.variant not in ("r2", "varsortability"):
            raise ValueError(f"variant must be one of 'r2' or 'varsortability', got {self.variant!r}.")
        if any(X.std() == 0):
            constant_cols = X.columns[X.std() == 0].tolist()
            raise ValueError(
                f"The following column(s) have zero variance (constant values): "
                f"{constant_cols}. Please drop these columns before fitting."
            )

        # clone the estimator or use default LinearRegression
        model_reg = clone(self.estimator) if self.estimator else LinearRegression()

        # Step 1: Generate the topological order.

        # Step 1.1: Iterate over nodes and compute the metric R^2 or variance.
        all_nodes_set = set(self.feature_names_in_)
        if self.variant == "r2":
            order_values = {}
            for target in self.feature_names_in_:
                other_nodes = list(all_nodes_set - {target})

                y = X[target]
                predictors = X[other_nodes]

                model_reg.fit(predictors, y)
                predictions = model_reg.predict(predictors)

                order_values[target] = r2_score(y, predictions)
        else:
            order_values = X.var().to_dict()

        # Step 1.2: Use metrics to get the topological order.
        sorted_nodes = sorted(order_values, key=order_values.get)

        # Step 2: Construct the DAG using the computed metrics.
        model = DAG()
        model.add_nodes_from(sorted_nodes)

        # Step 2.1: Adaptive Lasso (BIC-selected penalty) for parent selection.
        for i in range(1, len(sorted_nodes)):
            target = sorted_nodes[i]
            potential_parents = sorted_nodes[:i]

            y = X[target].to_numpy().ravel()
            predictors = X[potential_parents].to_numpy()

            # Step 2.1.1: least-squares fit supplies the adaptive weights.
            model_reg.fit(predictors, y)
            weights = np.abs(model_reg.coef_)

            # Step 2.1.2: L1 penalty on the reweighted design, lambda chosen by BIC.
            # Reweighting makes the selection invariant to rescaling of columns.
            sparse_reg = LassoLarsIC(criterion="bic")
            sparse_reg.fit(predictors * weights, y)
            coefs = sparse_reg.coef_ * weights

            for idx, coef in enumerate(coefs):
                if coef != 0:
                    model.add_edge(potential_parents[idx], target)

        self.causal_graph_ = model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, nodelist=self.feature_names_in_, weight=None, dtype="int"
        )

        return self
