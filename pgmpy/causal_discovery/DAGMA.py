import math
from functools import partial

import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from scipy.special import expit as sigmoid
from torch.optim import LBFGS
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
<<<<<<< HEAD
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.estimators import ExpertKnowledge
>>>>>>> origin/feature/dagma
from pgmpy.global_vars import logger
from pgmpy.utils import compat_fns


class DagmaLinear(_BaseCausalDiscovery):
    """
    DagmaLinear is a continuous optimization algorithm for causal discovery.

    It learns a Directed Acyclic Graph (DAG) from observational data by
    optimizing a continuous score function (Least Squares) subject to a
    novel log-determinant acyclicity constraint.

    Parameters :

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

    Attributes :

    ------------

    causal_graph_ : pgmpy.base.DAG
        The estimated Directed Acyclic Graph.

    adjacency_matrix_ : np.ndarray
        The estimated weighted adjacency matrix.

    n_features_in_ : int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the dataset used to learn the causal graph.

    Examples :

    ---------

    >>> from pgmpy.causal_discovery import DagmaLinear
    >>> ...

    References :

    ----------

    DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.
    Kevin Bello, Bryon Aragam, Pradeep Ravikumar.
    Booth School of Business, University of Chicago, Chicago, IL 60637.
    Machine Learning Department, Carnegie Mellon University, Pittsburgh, PA 15213

    """

    def __init__(
        self,
        s=1.0,
        lambda1=0.05,
        mu_init=1.0,
        mu_factor=0.1,
        max_iter=100,
        w_threshold=0.3,
        expert_knowledge=None,
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
        self.expert_knowledge = expert_knowledge

    def _fit(self, X: pd.DataFrame):
        """
        Core flow of the DAGMA continuous optimization algorithm.
        """
        # Step 1: Initialize states and extract feature dimensions
        self.feature_names_in_ = X.columns.values
        self.n_features_in_ = len(self.feature_names_in_)

        if self.n_features_in_ < 2:
            raise ValueError(
                f"Found array with 1 feature(s) while a minimum of 2 is required for causal dicovery"
            )

        # Step 2: Data Preparation & Covariance pre-computation
        data_np = X.values
        data_np = data_np - np.mean(data_np, axis=0, keepdims=True)
        self.cov_ = (data_np.T @ data_np) / float(data_np.shape[0])

        # Step 2.5: Build the Expert Knowledge Mask
        # Initialize a mask of all 1s (all edges allowed)
        mask = np.ones((self.n_features_in_, self.n_features_in_))

        if self.expert_knowledge is not None:
            self.expert_knowledge._orient_temporal_forbidden_edges(DAG(), only_edges=False)

            # Map forbidden edges into the binary mask
            feature_list = list(self.feature_names_in_)
            for u, v in self.expert_knowledge.forbidden_edges:
                try:
                    i = feature_list.index(u)
                    j = feature_list.index(v)
                    mask[i, j] = 0.0  # Set the forbidden edge's mask value to 0
                except ValueError:
                    continue  # Ignore if nodes provided in expert knowledge are not in the datas

                if u in feature_list and v in feature_list:
                    i = feature_list.index(u)
                    j = feature_list.index(v)
                    mask[i, j] = 0.0  # Set the forbidden edge's mask value to 0

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
            res = sopt.minimize(
                fun=self._objective,
                x0=W_est.flatten(),
                args=(mu, mask),     # PASS THE MASK TO THE OBJECTIVE FUNCTION
                method="L-BFGS-B",
                jac=True,
                bounds=bounds
            )
            W_est = res.x.reshape(self.n_features_in_, self.n_features_in_)

            # Ensure forbidden edges remain strictly zero in the main matrix
            W_est = W_est * mask
            mu *= self.mu_factor

        # Step 5: Thresholding and Graph Creation
        W_est[np.abs(W_est) < self.w_threshold] = 0
        self.adjacency_matrix_ = W_est

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(self.feature_names_in_)

        edges = np.argwhere(W_est != 0)
        for i, j in edges:
            self.causal_graph_.add_edge(self.feature_names_in_[i], self.feature_names_in_[j])

        return self

    def _objective(self, w_1d: np.ndarray, mu: float, mask : np.ndarray) -> tuple[float, np.ndarray]:
        """
        The objective function for the continuous optimization, combining
        the Least Squares score, L1 penalty, and Log-Det acyclicity constraint.
        """

        # Step 1: Reshape the flat 1D array back into a 2D adjacency matrix
        W = w_1d.reshape(self.n_features_in_, self.n_features_in_)

        # EXPERT KNOWLEDGE: Strictly enforce forbidden edges to be 0 in W
        W = W * mask

        # Step 2: Compute the Least Squares loss and gradient (Covariance Trick)
        dif = np.eye(self.n_features_in_) - W
        rhs = self.cov_ @ dif
        score = 0.5 * np.trace(dif.T @ rhs)
        G_score = -rhs

        # Step 3: Compute the Log-Determinant Acyclicity Constraint and gradient
        M = self.s * np.eye(self.n_features_in_) - (W * W)
        sign, logdet = np.linalg.slogdet(M)

        # Barrier Protection: Force backtrack if we step out of the valid DAG domain
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

        # EXPERT KNOWLEDGE MASKING:
        # Zero out the gradients of forbidden edges so the optimizer
        # never increase their weights
        G_obj = G_obj * mask

        # SciPy expects a flat 1D gradient array
        return obj, G_obj.flatten()

