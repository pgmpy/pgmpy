from __future__ import annotations

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy.causal_discovery._base import BaseCausalDiscovery, _BaseDAGMAMixin

torch = _safe_import("torch")


class DAGMALinear(_BaseDAGMAMixin, BaseCausalDiscovery):
    r"""
    DAGMA is a continuous optimization algorithm for causal discovery.

    It learns a Directed Acyclic Graph (DAG) from observational data by optimizing a continuous score function
    (Least Squares) subject to a novel log-determinant acyclicity constraint.

    Unlike older methods that rely on the Augmented Lagrangian scheme, DAGMA uses a central path method. It solves a
    sequence of unconstrained optimization problems where a central path parameter :math:`\mu` is progressively decayed.
    As :math:`\mu` approaches zero, the solution is mathematically guaranteed to be a DAG.

    The exact continuous optimization objective being minimized is:

    .. math::
        \min_{W} \mu \cdot (Q(W; X) + \lambda_1 \|W\|_1) + h(W)

    Where:
        - :math:`\mu` is the central path parameter.
        - :math:`Q(W; X) = \frac{1}{2n} \|X - XW\|_F^2` (Least Squares Loss)
        - :math:`h(W) = -\log \det(sI - W \circ W) + d \log s` (Log-Det Acyclicity Constraint)
        - :math:`\|W\|_1` is the L1 penalty to enforce sparsity.
        - :math:`\lambda_1` is L1 regularization coefficient.

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

    max_iter : int or None, optional (default=None)
        Maximum number of outer iterations for the central path optimization.
        Each iteration performs multiple inner optimization steps (controlled by ``inner_iter``)
        on the current mu value. If ``None``, defaults to 5 for Adam and 100 for L-BFGS,
        matching the official DAGMA implementation.

    inner_iter : int or None, optional (default=None)
        Number of inner optimization steps to perform per outer iteration (mu level).
        If ``None``, defaults to 3000 for Adam (which requires many steps to converge at
        each mu level) and 1 for L-BFGS (which performs internal line search).

    w_threshold : float, optional (default=0.3)
        Threshold for pruning small edge weights in the final adjacency matrix. Edges with absolute weight less than
        this threshold are set to zero. Higher values (e.g., 0.5) produce sparser graphs. Lower values (e.g., 0.1)
        retain more edges.

    return_type : str, optional (default="dag")
        The type of graph to return. Must be either "dag" or "cpdag".

    optimizer : type or None, optional (default=None)
        An uninstantiated PyTorch optimizer class (e.g., ``torch.optim.Adam``, ``torch.optim.LBFGS``).
        If ``None``, defaults to ``torch.optim.Adam`` to match the official DAGMA implementation.
        Any ``torch.optim.Optimizer`` subclass is accepted.

    optimizer_kwargs : dict or None, optional (default=None)
        Keyword arguments passed to the optimizer constructor. If ``None``, sensible defaults
        are used: ``{"lr": 0.0002}`` for Adam, or the optimizer's PyTorch defaults for other
        optimizers. For ``torch.optim.LBFGS`` a recommended configuration is
        ``{"max_iter": 10, "line_search_fn": "strong_wolfe"}``.

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
    >>> from pgmpy.causal_discovery import DAGMALinear
    >>> from pgmpy.datasets import load_dataset
    >>> # Load the Sachs continuous dataset
    >>> data = load_dataset("sachs_continuous").data
    >>> # Learn the causal structure
    >>> est = DAGMALinear()
    >>> est.fit(data)  # doctest: +SKIP
    >>> print(list(est.causal_graph_.edges()))  # doctest: +SKIP

    References
    ----------
    :cite:p:`bello_aragam_ravikumar_2020`
    """

    def __init__(
        self,
        s=1.0,
        lambda1=0.05,
        mu_init=1.0,
        mu_factor=0.1,
        max_iter=None,
        inner_iter=None,
        w_threshold=0.3,
        return_type: str = "dag",
        optimizer=None,
        optimizer_kwargs=None,
    ) -> None:
        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def _fit(self, X: pd.DataFrame):
        r"""
        Core flow of the DAGMA continuous optimization algorithm.

        The algorithm uses a central path method that optimizes a sequence of unconstrained problems. As
        :math:`\mu` decays to zero, the solution converges to a DAG.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.
        """
        # Step 1: Resolve device & dtype
        device, dtype = self._resolve_device_and_dtype()

        # Step 2: Pre-compute covariance matrix
        data_np = X.values
        data_np = data_np - np.mean(data_np, axis=0, keepdims=True)
        cov = (data_np.T @ data_np) / float(data_np.shape[0])
        cov_tensor = torch.tensor(cov, device=device, dtype=dtype)

        # Step 3: Initialize the weight matrix
        W_tensor = torch.zeros((self.n_features_in_, self.n_features_in_), device=device, dtype=dtype)

        # This allows _optimize to be reused by DAGMANonlinear which has a different objective signature without
        # coupling the mixin to any specific objective.
        def objective_fn(W, mu):
            return self._objective(W, mu, cov_tensor)

        # Resolve optimizer: None → Adam (matches official DAGMA package)
        optimizer_cls = self.optimizer if self.optimizer is not None else torch.optim.Adam

        # Resolve kwargs: None → sensible defaults per optimizer
        if self.optimizer_kwargs is not None:
            opt_kwargs = self.optimizer_kwargs
        elif optimizer_cls is torch.optim.Adam:
            opt_kwargs = {"lr": 0.0002}
        else:
            opt_kwargs = {}

        # Resolve iterations based on optimizer type
        max_iter_val = self.max_iter
        inner_iter_val = self.inner_iter

        if optimizer_cls is torch.optim.Adam:
            max_iter_val = 5 if max_iter_val is None else max_iter_val
            inner_iter_val = 3000 if inner_iter_val is None else inner_iter_val
        else:
            # Defaults for L-BFGS and others
            max_iter_val = 100 if max_iter_val is None else max_iter_val
            inner_iter_val = 1 if inner_iter_val is None else inner_iter_val

        # Step 4: Central Path Optimization Loop (from mixin)
        W_est_final = self._optimize(
            W_tensor=W_tensor,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=opt_kwargs,
            objective_fn=objective_fn,
            mu_init=self.mu_init,
            mu_factor=self.mu_factor,
            max_iter=max_iter_val,
            inner_iter=inner_iter_val,
        )

        self.adjacency_matrix_ = W_est_final

        # Step 5 & 6: Threshold and Convert to pgmpy DAG (from mixin)
        self.causal_graph_ = self._convert_to_dag(
            W_est_final, list(self.feature_names_in_), self.w_threshold, self.return_type
        )

        return self

    def _objective(self, W: torch.Tensor, mu: float, cov: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the DAGMA objective function.

        .. math::
            \text{obj}(W) = \mu \cdot (Q(W; X) + \lambda_1 \|W\|_1) + h(W)

        The objective combines three components:

        1. Least Squares loss: :math:`Q(W; X) = 0.5 \cdot \text{tr}((I - W)^T \hat{\Sigma} (I - W))`
        2. L1 penalty: :math:`\lambda_1 \|W\|_1`
        3. Log-Det barrier: :math:`h(W) = -\log \det(sI - W \circ W) + d \log s`

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

        # h(W) = -log det(sI - W ∘ W) + d·log(s)
        is_cyclic, h = self._log_det_barrier(W, self.s)

        # Barrier protection: return large finite loss to force backtracking
        if is_cyclic:
            return self.lambda1 * torch.abs(W).sum() + 1e10

        # Q(W; X) = 0.5 · tr((I - W)^T Σ̂ (I - W))
        score = 0.5 * torch.trace((eye - W).T @ cov @ (eye - W))

        # obj = μ · (Q + λ₁‖W‖₁) + h(W)
        obj = mu * (score + self.lambda1 * torch.abs(W).sum()) + h

        return obj
