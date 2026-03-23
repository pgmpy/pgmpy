import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression

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
    fast_ica: sklearn.decomposition.FastICA
        An instance of FastICA to use for independent component analysis. If None,
        a default FastICA instance with `max_iter=1000` is used.

    alpha: float, default=0.05
        Significance level for the Wald test used to prune edges.

    return_type: str, default="dag"
        The type of graph to return. Currently only "dag" is supported.

    Attributes
    ----------
    causal_graph_: pgmpy.base.DAG
        The learned causal graph.

    adjacency_matrix_: numpy.ndarray
        The learned adjacency matrix of the graph. Elements correspond to coefficients
        in the linear model.

    n_features_in_: int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_: list
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
        alpha: float = 0.05,
        return_type: str = "dag",
    ):
        self.alpha = alpha
        self.return_type = return_type
        self.fast_ica = fast_ica

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

        B_tilde = self._prune_edges(X_vals, B_hat, causal_order, alpha=self.alpha)

        self.adjacency_matrix_ = B_tilde

        # Step 6: Construct graph
        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(self.feature_names_in_)

        # Step 6: Add edges to the graph
        for target_idx in range(n_features):
            for source_idx in range(n_features):
                if B_tilde[target_idx, source_idx] != 0:
                    source_name = self.feature_names_in_[source_idx]
                    target_name = self.feature_names_in_[target_idx]

                    self.causal_graph_.add_edge(source_name, target_name)

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

    def _prune_edges(self, X: np.ndarray, B_hat: np.ndarray, causal_order: list, alpha: float = 0.01) -> np.ndarray:
        """
        Perform a Wald test to prune statistically insignificant edges from the
        estimated LiNGAM connection matrix. This follows the straightforward pruning
        approach outlined in Section 6.1 of the paper.

        Parameters
        ----------
        X : np.ndarray
            The observed dataset array of shape (n_samples, n_features).
        B_hat : np.ndarray
            The fully connected, lower-triangular estimated connection matrix from ICA.
        causal_order : list
            The topological causal ordering of the variables. Nodes can only be caused
            by nodes earlier in this list.
        alpha : float
            The significance level for the Wald test p-value. Edges with
            p-values >= alpha are pruned (set to 0).

        Returns
        -------
        B_pruned : np.ndarray
            The pruned adjacency matrix with insignificant edges removed.
        """

        n_samples, n_features = X.shape
        B_pruned = np.zeros_like(B_hat)

        for i, target_node in enumerate(causal_order):
            potential_parents = causal_order[:i]
            if len(potential_parents) == 0:
                continue

            # Extract the target variable and the feature matrix of its potential parents
            y_target = X[:, target_node]
            X_parents = X[:, potential_parents]

            # Fit OLS regression: y = X_parents * beta + error
            reg = LinearRegression().fit(X_parents, y_target)
            coefs = reg.coef_

            # Regression residuals to estimate the error variance
            y_pred = reg.predict(X_parents)
            residuals = y_target - y_pred

            # Estimate the variance of the residuals (sigma^2)
            # Use degrees of freedom (ddof) correction if there are more samples than
            # parent features
            sigma_sq = (
                np.var(residuals, ddof=len(potential_parents))
                if len(residuals) > len(potential_parents)
                else np.var(residuals)
            )

            # Covariance matrix of the regression coefficients
            # var(beta) = sigma^2 * (X^T * X)^-1
            XtX_inv = np.linalg.pinv(X_parents.T @ X_parents)
            var_beta = sigma_sq * XtX_inv

            standard_errors = np.sqrt(np.diag(var_beta))

            # Perform the Wald test for each potential parent
            for j, parent_node in enumerate(potential_parents):
                if standard_errors[j] > 0:
                    # The Wald statistic is (beta_hat / SE(beta_hat))^2
                    # Under the null hypothesis (true beta = 0), this follows a
                    # chi-square distribution with 1 DOF.
                    wald_stat = (coefs[j] ** 2) / (standard_errors[j] ** 2)
                    p_value = 1 - chi2.cdf(wald_stat, df=1)
                else:
                    p_value = 0

                if p_value < alpha:
                    B_pruned[target_node, parent_node] = coefs[j]

        return B_pruned
