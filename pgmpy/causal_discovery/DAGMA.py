from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery

torch = _safe_import("torch")
LBFGS = torch.optim.LBFGS


class DAGMALinear(_BaseCausalDiscovery):
    """
    DAGMA is a continuous optimization algorithm for causal discovery.

    It learns a Directed Acyclic Graph (DAG) from observational data by optimizing a continuous score function
    (Least Squares) subject to a novel log-determinant acyclicity constraint.

    Unlike older methods that rely on the Augmented Lagrangian scheme, DAGMA uses a central path method. It solves a
    sequence of unconstrained optimization problems where a central path parameter `mu` is progressively decayed.
    As `mu` approaches zero, the solution is mathematically guaranteedto be a DAG.

    The exact continuous optimization objective being minimized is:

    .. math::
        min_{W} \\mu \\cdot (Q(W; X) + \\lambda_1 \\|W\\|_1) + h(W)

    Where:
        - ``mu`` is the central path parameter.
        - ``Q(W; X) = 1/(2n) \\cdot \\|X - XW\\|_F^2`` (Least Squares Loss)
        - ``h(W) = -\\log \\det(sI - W \\circ W) + d \\log s``
          (Log-Det Acyclicity Constraint)
        - ``\\|W\\|_1`` is the L1 penalty to enforce sparsity.
        - ``lambda1`` is L1 regularization (enforce sparsity)

    Parameters
    ----------
    s : float, optional (default=1.0)
        Controls the domain of the M-matrices for the log-det constraint.
        Higher values (e.g., 2.0) make the acyclicity constraint more permissive, potentially allowing denser graphs.
        Lower values (e.g., 0.5) make it stricter, encouraging sparser solutions.

    lambda1 : float, optional (default=0.05)
        L1 regularization coefficient to enforce sparsity in the estimated graph.
        Higher values (e.g., 0.1) promote sparsity by shrinking edge weights to zero. Low values (e.g.,0.01) allow more
        edges in the estimated structure.

    mu_init : float, optional (default=1.0)
        Initial penalty parameter for the central path barrier method.
        Controls how strongly the acyclicity constraint is enforced at the start of optimization. Higher values start
        more aggressively enforcing acyclicity.

    mu_factor : float, optional (default=0.1)
        Decay factor for the penalty parameter.
        After each outer iteration, mu is multiplied by this factor (mu *= mu_factor). Smaller values (e.g., 0.01)
        decay faster, reaching the DAG constraint sooner but with potentially less optimization of the least squares
        fit. Larger values (e.g., 0.5) decay slower, allowing more fitting iterations but requiring more outer loops.

    max_iter : int, optional (default=100)
        Maximum number of iterations for the central path optimization.
        Each iteration performs one L-BFGS optimization step on the current mu value. More iterations allow better
        convergence but increase computation time.

    w_threshold : float, optional (default=0.3)
        Threshold for pruning small edge weights in the final adjacency matrix. Edges with absolute weight less than
        this threshold are set to zero. Higher values (e.g., 0.5) produce sparser graphs. Lower values (e.g., 0.1)
        retain more edges.

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
    Load a continuous dataset and discover causal structure:

    >>> from pgmpy.causal_discovery import DAGMALinear
    >>> from pgmpy.datasets import load_dataset

    Load the Sachs continuous dataset:

    >>> data = load_dataset("sachs_continuous").data

    Learn the causal structure:

    >>> est = DAGMALinear()
    >>> est.fit(data)
    >>> print(list(est.causal_graph_.edges()))

    >>> Output: [('raf', 'mek'), ('plc', 'pip2'), ('erk', 'akt'), ('erk', 'pka'),('akt', 'pka'), ('pkc', 'p38'),
    ('pkc', 'jnk'), ('jnk', 'p38')]

    References
    ----------
    .. [1] DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.
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
        Initialize the DAGMALinear estimator with hyperparameters
        """

        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.w_threshold = w_threshold

    def _fit(self, X: pd.DataFrame):
        """
        Core flow of the DAGMA continuous optimization algorithm.

        The algorithm uses a central path method that optimizes a sequence of unconstrained problems. As mu decays to
        zero, the solution converges to a DAG.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.
        """
        # Step 1: Resolve device and dtype from pgmpy config
        # This allows the algorithm to run on GPU if available
        device = config.get_device()
        dtype = config.get_dtype()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Step 2: Pre-compute covariance matrix
        data_np = X.values
        data_np = data_np - np.mean(data_np, axis=0, keepdims=True)
        cov = (data_np.T @ data_np) / float(data_np.shape[0] - 1)
        cov_tensor = torch.tensor(cov, device=device, dtype=dtype)

        # Step 3: Initialize the weight matrix and central path parameter
        W_est = np.zeros((self.n_features_in_, self.n_features_in_))
        mu = self.mu_init

        # Step 4: Central Path Optimization Loop
        for i in range(self.max_iter):
            # Create PyTorch parameter for current W estimate
            W_tensor = torch.nn.Parameter(torch.from_numpy(W_est).to(device=device, dtype=dtype))

            # Initialize L-BFGS optimizer
            optimizer = LBFGS([W_tensor], max_iter=10, line_search_fn="strong_wolfe")

            def closure(W=W_tensor):
                optimizer.zero_grad()
                loss = self._objective(W, mu, cov_tensor)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Extract updated W for next iteration
            W_est = W_tensor.detach().cpu().numpy()
            # Decay mu to tighten acyclicity constraint
            mu *= self.mu_factor

        # Step 5: Post-processing
        W_est[np.abs(W_est) < self.w_threshold] = 0
        self.adjacency_matrix_ = W_est

        # Step 6: Convert to pgmpy DAG object
        df_adj = pd.DataFrame(W_est.astype(np.float64), index=self.feature_names_in_, columns=self.feature_names_in_)
        # Convert to NetworkX DiGraph, then to pgmpy's DAG wrapper
        nx_graph = nx.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
        self.causal_graph_ = DAG(nx_graph)

        return self

    def _objective(self, W: torch.Tensor, mu: float, cov: torch.Tensor) -> torch.Tensor:
        """
        Compute the DAGMA objective function.

        The objective combines three components:
        1. Least Squares loss: Measures how well W explains the data
        2. L1 penalty: Promotes sparsity in the estimated graph
        3. Log-Det barrier: Enforces acyclicity via the central path method

        Parameters
        ----------
        W : torch.Tensor
            The adjacency matrix as a PyTorch tensor.
        mu : float
            Central path parameter. Controls the strength of the acyclicity constraint relative to the data fit.
        cov : torch.Tensor
            Pre-computed covariance matrix as a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            The objective value.
        """
        n = self.n_features_in_
        eye = torch.eye(n, dtype=W.dtype, device=W.device)

        # Component 1: Least Squares Score
        dif = eye - W
        rhs = cov @ dif
        score = 0.5 * torch.trace(dif.T @ rhs)

        # Component 2: Log-Determinant Acyclicity Constraint
        M = self.s * eye - (W * W)
        sign, logdet = torch.linalg.slogdet(M)

        # Barrier Protection: If it step outside the valid M-matrix domain, return a large finite loss to force the
        # optimizer to backtrack.
        if sign <= 0:
            # Return large loss while maintaining computation graph
            h = mu * (self.lambda1 * torch.abs(W).sum() + 1e10)
            return h

        h = -logdet + n * math.log(self.s)

        # Component 3: L1 Penalty for Sparsity
        l1_penalty = self.lambda1 * torch.abs(W).sum()

        # Combined Objective: Central Path formulation
        # As mu -> 0, the h(W) term dominates, enforcing acyclicity
        # As mu -> inf, the (score + l1_penalty) term dominates, fitting the data
        obj = mu * (score + l1_penalty) + h

        return obj
