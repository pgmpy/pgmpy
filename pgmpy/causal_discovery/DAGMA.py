import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from torch.optim import LBFGS

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.utils import compat_fns


class DagmaLinear(_BaseCausalDiscovery):
    """
    DagmaLinear is a continuous optimization algorithm for causal discovery.

    It learns a Directed Acyclic Graph (DAG) from observational data by
    optimizing a continuous score function (Least Squares) subject to a
    novel log-determinant acyclicity constraint.

    Unlike older methods that rely on the Augmented Lagrangian scheme, DAGMA
    uses a central path method. It solves a sequence of unconstrained
    optimization problems where a central path parameter `mu` is progressively
    decayed. As `mu` approaches zero, the solution is mathematically guaranteed
    to be a DAG.

    DAGMA is continuos optimization method that:
    1. Starts by initializing a continuous weighted adjacency matrix W
       with all zeros.
    2. Evaluates the exact mathematical gradient of the entire graph
       simultaneously.
    3. Uses a gradient-based numerical solver (like L-BFGS-B or Adam)
       to update all edge weights in the matrix W.
    4. It wraps the continuous optimizer in a loop that progressively decays
       the central path parameter μ.
    5. As μ approaches zero, the force of the log-determinant barrier becomes
       absolute, mathematically guaranteeing that the continuous matrix W
       converges to a perfect Directed Acyclic Graph.

    The exact continuous optimization objective being minimized is:
    min_{W}  mu * (Q(W; X) + lambda1 * ||W||_1) + h(W)

    Where:
        - mu is the central path parameter.
        - Q(W; X) = 1/(2n) * ||X - XW||_F^2  (Least Squares Loss)
        - h(W) = -log det(sI - W ∘ W) + d * log(s)
                    (Log-Det Acyclicity Constraint)
        - ||W||_1 is the L1 penalty to enforce sparsity.
        - lambda1 is L1 regularization (enforce sparsity)

    Parameters
    ----------
    s : float, optional (default=1.0)
        Controls the domain of the M-matrices for the log-det constraint.

    lambda1 : float, optional (default=0.05)
        L1 regularization coefficient to enforce sparsity in the estimated
        graph.

    mu_init : float, optional (default=1.0)
        Initial penalty parameter for the central path barrier method.

    mu_factor : float, optional (default=0.1)
        Decay factor for the penalty parameter.

    max_iter : int, optional (default=100)
        Maximum number of iterations for the central path optimization.

    w_threshold : float, optional (default=0.3)
        Threshold for pruning small edge weights in the final adjacency matrix.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The estimated Directed Acyclic Graph.

    adjacency_matrix_ : np.ndarray
        The estimated weighted adjacency matrix.

    n_features_in_ : int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the dataset used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/alarm")
    >>> data = model.simulate(n_samples=1000, seed=42)

    Or create a random data with causal relation

    >>> import numpy as np
    >>> import pandas as pd
    >>> data = pd.DataFrame(np.random.normal(size=(1000, 3)),
    >>>                     columns=['X', 'Y', 'Z'])
    >>> data['Y'] += 2.0 * data['X']
    >>> data['Z'] += 1.5 * data['Y']

    Use DagmaLinear algorith to learn causal structure

    >>> from pgmpy.causal_discovery import DagmaLinear
    >>> est = DagmaLinear()
    >>> est.fit(data)
    >>> est.causal_graph_.edges()

    References
    ----------
    .. [1] DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity
           Characterization.
           Kevin Bello, Bryon Aragam, Pradeep Ravikumar.
           Booth School of Business, University of Chicago, Chicago, IL 60637.
           Machine Learning Department, Carnegie Mellon University,
           Pittsburgh, PA 15213
    """

    def __init__(
        self,
        s=1.0,
        lambda1=0.05,
        mu_init=1.0,
        mu_factor=0.1,
        max_iter=100,
        w_threshold=0.3,
    ):
        """
        Initialize the DagmaLinear estimator with hyperparameters
        """

        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.backend = compat_fns.get_compute_backend()

    def _fit(self, X: pd.DataFrame):
        """
        Core flow of the DAGMA continuous optimization algorithm.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to learn the causal structure from.
        """
        # Step 1: Initialize states and extract feature dimensions
        self.feature_names_in_ = X.columns.values
        self.n_features_in_ = len(self.feature_names_in_)

        if self.n_features_in_ < 2:
            raise ValueError(
                f"Found array with 1 feature {self.feature_names_in_} while a"
                "minimum of 2 is required for causal dicovery"
            )

        # Step 2: Data Preparation & Covariance pre-computation
        data_np = X.values
        data_np = data_np - np.mean(data_np, axis=0, keepdims=True)
        self.cov_ = (data_np.T @ data_np) / float(data_np.shape[0])

        # Step 3: Configure bounds to strictly prevent self-loops
        bounds = [
            (0, 0) if i == j else (None, None)
            for i in range(self.n_features_in_)
            for j in range(self.n_features_in_)
        ]

        # Step 4: The Central Path Optimization Loop
        W_est = np.zeros((self.n_features_in_, self.n_features_in_))
        mu = self.mu_init

        for _ in range(self.max_iter):

            if self.backend == np:
                res = sopt.minimize(
                    fun=self._objective,
                    x0=W_est.flatten(),
                    args=(mu),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds
                 )
                W_est = res.x.reshape(self.n_features_in_, self.n_features_in_)
            else:  # Pytorch
                # Convert W_est to a PyTorch parameter
                W_tensor = torch.nn.Parameter(
                    torch.tensor(W_est,
                                 dtype=torch.float64,
                                 requires_grad=True)
                )

                # Initialize the PyTorch LBFGS optimizer
                lbfgs = LBFGS([W_tensor],
                              max_iter=5,
                              line_search_fn="strong_wolfe")

                def closure():
                    lbfgs.zero_grad()  # Clear previous gradients
                    loss = self._objective(W_tensor, mu)
                    loss.backward()  # Automatically computes the gradients
                    return loss

                # Take an optimization step
                lbfgs.step(closure)

                # Extract the updated numpy array for the next loop
                W_est = W_tensor.detach().numpy()

            mu *= self.mu_factor

        # Step 5: Thresholding and Graph Creation
        W_est[np.abs(W_est) < self.w_threshold] = 0
        self.adjacency_matrix_ = W_est

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(self.feature_names_in_)

        edges = np.argwhere(W_est != 0)
        for i, j in edges:
            self.causal_graph_.add_edge(self.feature_names_in_[i],
                                        self.feature_names_in_[j])

        return self

    def _objective(self,
                   w_1d: np.ndarray,
                   mu: float,
                   ) -> tuple[float, np.ndarray]:
        """
        The objective function for the continuous optimization, combining
        the Least Squares score, L1 penalty, and Log-Det acyclicity constraint.
        """

        if self.backend == np:
            # Step 1: Reshape the flat 1D array back into a 2D adjacency matrix
            W = w_1d.reshape(self.n_features_in_, self.n_features_in_)

            # Step 2: Compute the Least Squares loss and gradient
            dif = np.eye(self.n_features_in_) - W
            rhs = self.cov_ @ dif
            score = 0.5 * np.trace(dif.T @ rhs)
            G_score = -rhs

            # Step 3: Compute the Log-Determinant Acyclicity Constraint
            # and gradient
            M = self.s * np.eye(self.n_features_in_) - (W * W)
            sign, logdet = np.linalg.slogdet(M)

            # Barrier Protection: Force backtrack if we step out
            # of the valid DAG domain
            if sign <= 0:
                return np.inf, np.zeros_like(w_1d)

            h = -logdet + self.n_features_in_ * np.log(self.s)
            M_inv = slin.inv(M)
            Grad_h = 2 * W * M_inv.T

            # Step 4: Combine into the final DAGMA Central Path Objective
            l1_penalty = self.lambda1 * np.abs(W).sum()
            obj = mu * (score + l1_penalty) + h

            # Step 5: Combine gradients
            # (np.sign(W) is the subgradient of the L1 norm)
            G_obj = mu * (G_score + self.lambda1 * np.sign(W)) + Grad_h

            # SciPy expects a flat 1D gradient array
            return obj, G_obj.flatten()
        else:  # TODO for pythorch
            return obj
