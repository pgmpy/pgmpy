import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
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
    5. **Edge Pruning**: Construct the causal adjacency matrix $\tilde{B}$
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

    random_state : int, default=None
        Seed for the random number generator used by the ICA algorithm. Ensures
        reproducibility across repeated algorithm runs. If a `fast_ica` instance
        is passed, this will override its existing random state.

    max_iter : int, default=None
        Maximum number of iterations for the FastICA algorithm. Defaults to 1000.
        If a `fast_ica` instance is passed, this will override its existing max_iter.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph.

    adjacency_matrix_ : pd.DataFrame
        The learned adjacency matrix of the graph. Elements correspond to coefficients
        in the linear model.

    n_features_in_ : int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_ : np.ndarray
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
        random_state: int | None = None,
        max_iter: int | None = None,
    ):
        self.fast_ica = fast_ica
        self.estimator = estimator
        self.gamma = gamma
        self.return_type = return_type
        self.random_state = random_state
        self.max_iter = max_iter

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
            raise NotImplementedError("Only return_type='dag' is supported.")

        try:
            X_vals = np.asarray(X.values, dtype=float)
        except ValueError as e:
            raise ValueError("All features must be numeric.") from e

        _, n_features = X_vals.shape

        # Step 1: Resolve internal estimators and apply ICA
        if self.estimator is None:
            self._estimator = LinearRegression()
        else:
            self._estimator = clone(self.estimator)

        if self.fast_ica is None:
            _max_iter = self.max_iter if self.max_iter is not None else 1000
            ica = FastICA(max_iter=_max_iter, random_state=self.random_state)
        else:
            ica = clone(self.fast_ica)
            if self.random_state is not None:
                ica.set_params(random_state=self.random_state)
            if self.max_iter is not None:
                ica.set_params(max_iter=self.max_iter)

        ica.fit(X_vals)
        W = ica.components_

        if W.shape[0] != n_features:
            raise ValueError(f"FastICA n_components must equal n_features (got {W.shape[0]} != {n_features}).")

        # Step 2: Find permutation of rows of W.
        epsilon = 1e-12
        cost_matrix = 1 / (np.abs(W) + epsilon)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        W_perm = np.zeros_like(W)
        W_perm[col_ind] = W[row_ind]

        # Step 3: Divide rows of permuted W by diagonal elements.
        diag_W_perm = np.diag(W_perm)
        if np.any(np.abs(diag_W_perm) < epsilon):
            raise ValueError("Near-zero diagonal elements in ICA permutation.")
        W_scaled = W_perm / diag_W_perm[:, np.newaxis]
        B_hat = np.eye(n_features) - W_scaled

        # Step 4: Find a causal order
        causal_order = self._causal_order(B_hat)

        # Step 5: Construct the causal adjacency matrix.
        if causal_order is None:
            raise ValueError("Graph contains unresolvable cycles.")

        B_tilde = self._prune_edges(X_vals, causal_order)
        self.adjacency_matrix_ = pd.DataFrame(B_tilde, index=self.feature_names_in_, columns=self.feature_names_in_)

        # Step 6: Construct graph
        self.causal_graph_ = nx.from_pandas_adjacency(self.adjacency_matrix_, create_using=DAG())

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
            zero_row_mask = np.all(np.isclose(B_hat, 0.0, atol=1e-8), axis=1)
            row_indices = np.where(zero_row_mask)[0]
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

    def _adaptive_lasso(self, X: np.ndarray, X_std: np.ndarray, predictors: list, target: int) -> np.ndarray:
        r"""
        Helper function implementing the Adaptive Lasso algorithm for edge pruning.

        .. math::
            \beta^*_{(n)} = \arg\min_{\beta} \left\| \mathbf{y} - \sum_{j=1}^p \mathbf{x}_j \beta_j \right\|^2_2
            + \lambda_n \sum_{j=1}^p w_j |\beta_j|

        Parameters
        ----------
        X : np.ndarray
            The input data matrix.

        X_std : np.ndarray
            The standardized input data matrix.

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

        # Step 1: Pruning with Adaptive Lasso
        # Step 1.1: Fit the estimator to the standardized data
        self._estimator.fit(X_std[:, predictors], X_std[:, target])

        # Compute standard adaptive lasso weights w_j = 1 / |beta_init,j|^gamma
        # using an epsilon floor to prevent division by zero.
        weight = 1.0 / np.power(np.maximum(np.abs(self._estimator.coef_), 1e-12), self.gamma)

        # Step 1.2: Fit the Lasso regression on predictors scaled by 1 / w.
        # If z_j = x_j / w_j and theta_j are the lasso coefficients on z_j,
        # then the adaptive lasso coefficients are beta_j = theta_j / w_j.
        lasso_reg = LassoLarsIC(criterion="bic")
        lasso_reg.fit(X_std[:, predictors] / weight, X_std[:, target])
        adaptive_coef = lasso_reg.coef_ / weight
        pruned_idx = np.abs(adaptive_coef) > 0.0

        # Step 2: Calculate coefficients of the original scale
        coef = np.zeros(lasso_reg.coef_.shape)
        if pruned_idx.sum() > 0:
            pred = np.array(predictors)
            self._estimator.fit(X[:, pred[pruned_idx]], X[:, target])
            coef[pruned_idx] = self._estimator.coef_

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

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        B_pruned = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        for i in range(1, len(causal_order)):
            target = causal_order[i]
            predictors = causal_order[:i]

            if len(predictors) == 0:
                continue

            B_pruned[predictors, target] = self._adaptive_lasso(X, X_std, predictors, target)

        return B_pruned
