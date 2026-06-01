import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.utils import preprocess_data


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

    The LiNGAM model assumes $x = Bx + e$, where $B$ is a weight matrix representing
    a DAG and $e$ is non-Gaussian noise. The algorithm uses ICA to estimate $x = Ae$
    (where $W = A^{-1}$), followed by a permutation and scaling of $W$ to recover
    the causal order. Finally, pruning is done using sparse regression to obtain a
    sparser structure.

    Parameters
    ----------
    ica : sklearn.decomposition.FastICA
        An instance of FastICA to use for independent component analysis. If None,
        a default FastICA instance with `max_iter=1000` is used.

    pruning_estimator : sklearn.linear_model._base.LinearModel
        An instance of a linear model to use for the Adaptive Lasso pruning step.
        If None, a default LinearRegression instance is used.

    gamma : float, default=1.0
        The exponent used to calculate the adaptive weights in the Adaptive Lasso.
        A higher value leads to a sparser graph by increasing the penalty on
        small coefficients, while a lower value makes the weights more uniform.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

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
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.causal_discovery import LiNGAM
    >>> df = load_dataset("sachs_continuous").data
    >>> lingam = LiNGAM()
    >>> lingam.fit(df)
    >>> # The causal graph contains the learned edges among signaling proteins.
    >>> len(lingam.causal_graph_.edges())
    42


    References
    ----------
    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen. A linear non-gaussian
        acyclic model for causal discovery. Journal of Machine Learning Research,
        7: 2003--2030, 2006.
    """

    def __init__(
        self,
        ica: FastICA | None = None,
        pruning_estimator=None,
        gamma: float = 1.0,
        return_type: str = "dag",
    ):
        self.ica = ica
        self.pruning_estimator = pruning_estimator
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
        if self.return_type not in ["dag", "pdag"]:
            raise ValueError("return_type must be either 'dag' or 'pdag'.")

        X, dtypes = preprocess_data(X)
        if any(dtype != "N" for dtype in dtypes.values()):
            raise ValueError("All features must be numeric.")

        X_vals = X.values.astype(float)

        # Step 1: Resolve internal estimators and apply ICA
        if self.pruning_estimator is None:
            self._pruning_estimator = LinearRegression()
        else:
            self._pruning_estimator = clone(self.pruning_estimator)

        if self.ica is None:
            ica = FastICA(max_iter=1000, random_state=42)
        else:
            ica = clone(self.ica)

        if (ica.n_components is not None) and (ica.n_components != self.n_features_in_):
            raise ValueError(
                f"FastICA n_components must equal n_features (got {ica.n_components} != {self.n_features_in_})."
            )

        ica.fit(X_vals)
        W = ica.components_

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
        B_hat = np.eye(self.n_features_in_) - W_scaled

        # Step 4: Find a causal order
        causal_order = self._causal_order(B_hat)

        # Step 5: Construct the causal adjacency matrix.
        if causal_order is None:
            raise ValueError(
                "Unable to determine a valid causal ordering (topological sort). "
                "The recovered weight matrix contains persistent cycles, suggesting "
                "that the DAG assumption is violated or the ICA estimation is "
                "insufficiently converged due to noise or inadequate sample size."
            )

        B_tilde = self._prune_edges(X_vals, causal_order)
        self.adjacency_matrix_ = pd.DataFrame(B_tilde, index=self.feature_names_in_, columns=self.feature_names_in_)

        # Step 6: Construct graph
        dag = nx.from_pandas_adjacency(self.adjacency_matrix_, create_using=DAG())
        if self.return_type == "pdag":
            self.causal_graph_ = PDAG(directed_ebunch=list(dag.edges()))
        else:
            self.causal_graph_ = dag

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
        self._pruning_estimator.fit(X_std[:, predictors], X_std[:, target])

        # Compute standard adaptive lasso weights w_j = 1 / |beta_init,j|^gamma
        # using an epsilon floor to prevent division by zero.
        weight = 1.0 / np.power(np.maximum(np.abs(self._pruning_estimator.coef_), 1e-12), self.gamma)

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
            self._pruning_estimator.fit(X[:, pred[pruned_idx]], X[:, target])
            coef[pruned_idx] = self._pruning_estimator.coef_

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
