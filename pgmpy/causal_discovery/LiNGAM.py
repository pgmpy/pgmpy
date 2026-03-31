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
    """
    LiNGAM (Linear Non-Gaussian Acyclic Model) finds the causal order under three assumptions:
    1. The causal graph is acyclic.
    2. The causal relationships are linear.
    3. The noise terms are non-Gaussian.
    A model with these three properties we call a Linear, Non-Gaussian, Acyclic Model,
    abbreviated LiNGAM.

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
    >>> from pgmpy.causal_discovery import LiNGAM
    >>> X = pd.DataFrame({"x1": [1, 2, 3], "x2": [2, 4, 6], "x3": [3, 6, 9]})
    >>> algo = LiNGAM()
    >>> algo.fit(X)

    References
    ----------
    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen. A linear non-gaussian
        acyclic model for causal discovery. Journal of Machine Learning Research,
        7: 2003--2030, 2006.
    """

    def __init__(
        self,
        fast_ica=None,
        estimator=None,
        gamma: float = 1.0,
        return_type: str = "dag",
    ):
        self.fast_ica = fast_ica
        self.gamma = gamma
        self.estimator = estimator if estimator is not None else LinearRegression()
        self.return_type = return_type

    def _fit(self, X: pd.DataFrame):

        # Step 0: Validate inputs
        if self.return_type != "dag":
            raise NotImplementedError(f"Return type {self.return_type} is not yet implemented. Use 'dag'.")

        X_vals = X.values
        n_samples, n_features = X_vals.shape
        self.n_features_in_ = n_features
        self.feature_names_in_ = list(X.columns)

        # Step 1: Apply an ICA algorithm to obtain a decomposition X = AS where S has
        # the same size as X and contains in its rows the independent components.
        # From here on, we will exclusively work with W = A^-1.
        if self.fast_ica is None:
            ica = FastICA(max_iter=1000)
        else:
            ica = self.fast_ica

        ica.fit(X_vals)
        W = ica.components_

        # Step 2: Find the one and only permutation of rows of W which yields a matrix
        # W_perm without any zeros on the main diagonal. In practice, small estimation
        # errors will cause all elements of W to be non-zero, and hence the permutation
        # is sought which minimizes sum_i 1/|W_perm_ii|.
        cost_matrix = 1 / np.abs(W)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        W_perm = np.zeros_like(W)
        W_perm[col_ind] = W[row_ind]

        # Step 3: Divide each row of W_perm by its corresponding diagonal element, to
        # yield a new matrix W_scaled with ones on the diagonal. Then, compute an
        # estimate B_hat of B using B_hat = I - W_scaled.
        W_scaled = W_perm / np.diag(W_perm)[:, np.newaxis]
        B_hat = np.eye(n_features) - W_scaled

        # Step 4: Find a causal order
        causal_order = self._causal_order(B_hat)

        # Step 5: Construct the lower triangular causal matrix B_tilde
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
        """Helper function for _causal_order to find a causal order from the given
        matrix strictly. Implements the Algorithm B from section 5.2 of the paper.

        Parameters
        ----------
        B_hat : np.ndarray
            Weight matrix obtained from ICA with in absolute value set to zero.

        Returns
        -------
        causal_order : list | None
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = []

        row_num = B_hat.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(B_hat):
            # find a row i such that all of which elements are zero
            row_index_list = np.where(np.sum(np.abs(B_hat), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove the i-th row and the i-th column from matrix
            mask = np.delete(np.arange(len(B_hat)), target_index, axis=0)
            B_hat = B_hat[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order

    def _causal_order(self, B_hat: np.ndarray) -> list | None:
        """Helper function to obtain a causal order from the given matrix approximately.
        Implements the Algorithm C from section 5.2 of the paper.

        Parameters
        ----------
        B_hat : np.ndarray
            Weight matrix obtained from ICA.

        Returns
        -------
        causal_order : list | None
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = None
        B_hat = B_hat.copy()

        # Step 1: Set the m(m + 1)/2 smallest(in absolute value) elements of B_hat to zero
        pos_list = np.argsort(np.abs(B_hat), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, B_hat.shape)).T
        initial_zero_num = int(B_hat.shape[0] * (B_hat.shape[0] + 1) / 2)

        for i, j in pos_list[:initial_zero_num]:
            B_hat[i, j] = 0

        causal_order = self._search_causal_order(B_hat)
        if causal_order is not None:
            return causal_order

        # Step 2: Test if B_hat can be permuted to a lower triangular matrix
        for i, j in pos_list[initial_zero_num:]:
            B_hat[i, j] = 0
            causal_order = self._search_causal_order(B_hat)
            if causal_order is not None:
                break

        return causal_order

    def _adaptive_lasso(self, X: np.ndarray, predictors: list, target: int) -> np.ndarray:
        r"""
        This is a helper function which implements the Adaptive Lasso algorithm.

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
        """

        # Step 1: Standardize X
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Step 2: Pruning with Adaptive Lasso
        # Step 2.1: Fit the estimator to the standardized data
        self.estimator.fit(X_std[:, predictors], X_std[:, target])
        weight = np.power(np.abs(self.estimator.coef_), self.gamma)

        # Step 2.2: Fit the Lasso regression to the weighted standardized data
        Lasso_reg = LassoLarsIC(criterion="bic")
        Lasso_reg.fit(X_std[:, predictors] * weight, X_std[:, target])
        pruned_idx = np.abs(Lasso_reg.coef_ * weight) > 0.0

        # Step 3: Calculate coefficients of the original scale
        coef = np.zeros(Lasso_reg.coef_.shape)
        if pruned_idx.sum() > 0:
            pred = np.array(predictors)
            self.estimator.fit(X[:, pred[pruned_idx]], X[:, target])
            coef[pruned_idx] = self.estimator.coef_

        return coef

    def _prune_edges(self, X: np.ndarray, causal_order: list) -> np.ndarray:
        """
        This function is used to prune the edges of the causal graph.
        It uses the Adaptive Lasso algorithm to prune the edges of the causal graph.

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

        References
        ----------
        Zou, H. (2006). The Adaptive Lasso and Its Oracle Properties. Journal of the American Statistical Association,
         101(476), 1418–1429.

        See Also
        --------
        pgmpy.causal_discovery.LiNGAM._adaptive_lasso
        """

        B_pruned = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(causal_order)):
            target = causal_order[i]
            predictors = causal_order[:i]

            if len(predictors) == 0:
                continue

            B_pruned[target, predictors] = self._adaptive_lasso(X, predictors, target)

        return B_pruned
