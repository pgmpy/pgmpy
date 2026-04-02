import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery


class LiNGAM(_BaseCausalDiscovery):
    r"""
    Continuous data causal discovery using the Linear Non-Gaussian Acyclic Model (LiNGAM).

    This class implements the LiNGAM algorithm [1]_ for causal discovery. Given a
    tabular dataset, the algorithm estimates the causal structure among the
    variables in the data as a Directed Acyclic Graph (DAG) by utilizing Independent
    Component Analysis (ICA).

    The algorithm relies on the following three core assumptions about the generated data:
    1. The true causal graph is a directed acyclic graph (no feedback loops).
    2. The causal relationships between variables are strictly linear.
    3. The residual error (noise) terms have a non-Gaussian distribution.

    A model with these three properties is called a Linear, Non-Gaussian, Acyclic
    Model, abbreviated LiNGAM.

    Algorithm
    ---------
    The LiNGAM algorithm estimates the causal structure using the following steps:
    1. **Independent Component Analysis (ICA)**: Apply ICA to the data matrix $X$ to
       obtain a decomposition $X = AS$, where $S$ contains the independent components
       in its rows. We then compute the unmixing matrix $W = A^{-1}$.
    2. **Row Permutation**: Find the unique row permutation of $W$ that yields a matrix
       $W_{perm}$ with no zeros on its main diagonal. To account for estimation errors,
       the optimal permutation is found by minimizing the cost function
       $\sum_{i} \frac{1}{|(W_{perm})_{ii}|}$.
    3. **Diagonal Scaling**: Normalize the rows of $W_{perm}$ by dividing each row
       by its corresponding diagonal element, resulting in a matrix $W_{scaled}$ with
       ones on the diagonal. Compute the connection strength matrix estimate as
       $\hat{B} = I - W_{scaled}$.
    4. **Causal Ordering**: Discover a valid causal ordering of the variables by
       recursively identifying and removing nodes with no parents from $\hat{B}$.
    5. **Edge Pruning**: Construct the lower triangular causal matrix $\tilde{B}$
       by applying sparse regression (Adaptive Lasso) to prune statistically
       insignificant edges based on the discovered causal ordering.

    Parameters
    ----------
    fast_ica : sklearn.decomposition.FastICA
        An instance of FastICA to use for independent component analysis. If None,
        a default FastICA instance with `max_iter=1000` is used.

    estimator : sklearn.linear_model._base.LinearModel
        An instance of a linear model to use for estimating the causal relationships.
        If None, a default LinearRegression instance is used.

    gamma : float, default=1.0
        The exponent used to calculate the adaptive weights in the Adaptive Lasso.

    return_type : str, default="dag"
        The type of graph to return. Currently only "dag" is supported.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph.

    adjacency_matrix_ : pd.DataFrame
        The learned adjacency matrix of the graph. Elements correspond to coefficients
        in the linear model.

    n_features_in_ : int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_ : list
        The feature names in the dataset used to learn the causal graph.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.causal_discovery import LiNGAM
    >>> X = pd.DataFrame(np.random.uniform(size=(100, 3)), columns=list("ABC"))
    >>> X["B"] = 2.0 * X["A"] + X["B"]
    >>> X["C"] = -1.5 * X["B"] + X["C"]
    >>> lingam = LiNGAM()
    >>> lingam.fit(X)
    >>> set(lingam.causal_graph_.edges())
    {('A', 'B'), ('B', 'C')}


    References
    ----------
    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen. A linear non-gaussian
        acyclic model for causal discovery. Journal of Machine Learning Research,
        7: 2003--2030, 2006.
    """

    def __init__(
        self,
        fast_ica: FastICA | None = None,
        estimator=None,
        gamma: float = 1.0,
        return_type: str = "dag",
    ):
        if fast_ica is None:
            self.fast_ica = FastICA(max_iter=1000)
        else:
            self.fast_ica = fast_ica

        if estimator is None:
            self.estimator = LinearRegression()
        else:
            self.estimator = estimator

        self.gamma = gamma
        self.return_type = return_type

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the LiNGAM algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset from which to learn the causal structure.

        Returns
        -------
        self : pgmpy.causal_discovery.LiNGAM
            Returns the instance with the fitted attributes.
        """

        # Step 0: Validate inputs
        if self.return_type != "dag":
            raise NotImplementedError(f"Return type {self.return_type} is not yet implemented. Use 'dag'.")

        X_vals = X.values
        n_samples, n_features = X_vals.shape
        self.n_features_in_ = n_features
        self.feature_names_in_ = list(X.columns)

        # Step 1: Apply an ICA algorithm to obtain a decomposition
        ica = self.fast_ica

        ica.fit(X_vals)
        W = ica.components_

        # Step 2: Find permutation of rows of W.
        cost_matrix = 1 / np.abs(W)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        W_perm = np.zeros_like(W)
        W_perm[col_ind] = W[row_ind]

        # Step 3: Divide rows of permuted W by diagonal elements.
        W_scaled = W_perm / np.diag(W_perm)[:, np.newaxis]
        B_hat = np.eye(n_features) - W_scaled

        # Step 4: Find a causal order
        causal_order = self._causal_order(B_hat)

        # Step 5: Construct the lower triangular causal matrix.
        if causal_order is None:
            raise ValueError("Could not find a valid causal order. Graph contains unresolvable cycles.")

        B_tilde = self._prune_edges(X_vals, causal_order)

        self.adjacency_matrix_ = pd.DataFrame(B_tilde, index=self.feature_names_in_, columns=self.feature_names_in_)

        # Step 6: Construct graph
        self.causal_graph_ = nx.from_numpy_array(B_tilde.T, create_using=DAG())
        nx.relabel_nodes(
            self.causal_graph_, mapping={i: name for i, name in enumerate(self.feature_names_in_)}, copy=False
        )

        return self

    def _search_causal_order(self, B_hat: np.ndarray) -> list | None:
        """
        Helper function to strictly determine a causal order from the given matrix.
        Implements Algorithm B from section 5.2 of the paper.

        Algorithm
        ---------
        1. **Identify Root Node**: Find a row index $i$ in the matrix where all
           elements are zero.
        2. **Append to Order**: Append $i$ to the end of the causal order list.
        3. **Matrix Reduction**: Remove the $i$-th row and $i$-th column from
           the matrix and repeat.

        Parameters
        ----------
        B_hat : np.ndarray
            Weight matrix obtained from ICA, where specific elements have been nullified.

        Returns
        -------
        causal_order : list | None
            A valid causal ordering of nodes if one exists, otherwise None.
        """
        causal_order = []

        n_rows = B_hat.shape[0]
        original_indices = np.arange(n_rows)

        while 0 < len(B_hat):
            # Step 1: Find an all-zero row.
            row_indices = np.where(np.sum(np.abs(B_hat), axis=1) == 0)[0]
            if len(row_indices) == 0:
                break

            target_index = row_indices[0]

            # Step 2: Append root node to causal order.
            causal_order.append(original_indices[target_index])
            original_indices = np.delete(original_indices, target_index, axis=0)

            # Step 3: Remove the row and column from the matrix.
            mask = np.delete(np.arange(len(B_hat)), target_index, axis=0)
            B_hat = B_hat[mask][:, mask]

        if len(causal_order) != n_rows:
            causal_order = None

        return causal_order

    def _causal_order(self, B_hat: np.ndarray) -> list | None:
        r"""
        Helper function to approximate a valid causal order from the given matrix.
        Implements Algorithm C from section 5.2 of the paper.

        Algorithm
        ---------
        1. **Nullify Minimal Elements**: Initially set the $m(m + 1)/2$ smallest
           (in absolute value) elements of the weight matrix $\hat{B}$ to zero.
        2. **Iterative Triangularization**: Sequentially set the next smallest
           elements to zero and verify if $\hat{B}$ can be permuted into a strictly
           lower triangular matrix using Algorithm B.

        Parameters
        ----------
        B_hat : np.ndarray
            Weight matrix obtained from ICA.

        Returns
        -------
        causal_order : list | None
            A valid causal ordering of nodes if one exists, otherwise None.
        """
        causal_order = None
        B_hat = B_hat.copy()

        # Step 1: Nullify minimal absolute elements.
        pos_list = np.argsort(np.abs(B_hat), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, B_hat.shape)).T
        initial_zero_num = int(B_hat.shape[0] * (B_hat.shape[0] + 1) / 2)

        for i, j in pos_list[:initial_zero_num]:
            B_hat[i, j] = 0

        causal_order = self._search_causal_order(B_hat)
        if causal_order is not None:
            return causal_order

        # Step 2: Iteratively nullify remaining elements to find a strictly lower triangular form.
        for i, j in pos_list[initial_zero_num:]:
            B_hat[i, j] = 0
            causal_order = self._search_causal_order(B_hat)
            if causal_order is not None:
                break

        return causal_order

    def _adaptive_lasso(self, X: np.ndarray, predictors: list, target: int) -> np.ndarray:
        r"""
        Helper function implementing the Adaptive Lasso algorithm for edge pruning.

        .. math::
            \beta^*_{(n)} = \arg\min_{\beta} \left\| \mathbf{y} - \sum_{j=1}^p \mathbf{x}_j \beta_j \right\|^2_2
            + \lambda_n \sum_{j=1}^p w_j |\beta_j|

        Parameters
        ----------
        X : np.ndarray
            The input data matrix.

        predictors : list
            The list of predictor variables.

        target : int
            The target variable.

        Returns
        -------
        coef : np.ndarray
            The pruned coefficients.


        References
        ----------
        .. [1] Zou, H. (2006). The adaptive lasso and its oracle properties.
               Journal of the American Statistical Association, 101(476), 1418–1429.
               https://doi.org/10.1198/016214506000000735
        """

        # Step 1: Standardize X
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Step 2: Pruning with Adaptive Lasso
        # Step 2.1: Fit the estimator to the standardized data
        self.estimator.fit(X_std[:, predictors], X_std[:, target])
        weight = np.power(np.abs(self.estimator.coef_), self.gamma)

        # Step 2.2: Fit the Lasso regression to the weighted standardized data
        lasso_reg = LassoLarsIC(criterion="bic")
        lasso_reg.fit(X_std[:, predictors] * weight, X_std[:, target])
        pruned_idx = np.abs(lasso_reg.coef_ * weight) > 0.0

        # Step 3: Calculate coefficients of the original scale
        coef = np.zeros(lasso_reg.coef_.shape)
        if pruned_idx.sum() > 0:
            pred = np.array(predictors)
            self.estimator.fit(X[:, pred[pruned_idx]], X[:, target])
            coef[pruned_idx] = self.estimator.coef_

        return coef

    def _prune_edges(self, X: np.ndarray, causal_order: list) -> np.ndarray:
        """
        Prunes insignificant edges from the causal graph by applying the Adaptive Lasso algorithm.

        Parameters
        ----------
        X : np.ndarray
            The input data matrix.
        causal_order : list
            The causal order of the given matrix.

        Returns
        -------
        B_pruned : np.ndarray
            The pruned causal matrix.

        See Also
        --------
        pgmpy.causal_discovery.LiNGAM._adaptive_lasso
        """

        B_pruned = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        for i in range(1, len(causal_order)):
            target = causal_order[i]
            predictors = causal_order[:i]

            if len(predictors) == 0:
                continue

            B_pruned[target, predictors] = self._adaptive_lasso(X, predictors, target)

        return B_pruned
