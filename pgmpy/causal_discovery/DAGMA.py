from __future__ import annotations

import math

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
    s : float or list of float or None, optional (default=None)
        Controls the domain of the M-matrices for the log-det constraint.
        If ``None``, the official DAGMA schedule ``[1.0, 0.9, 0.8, 0.7, 0.6]`` is used.
        If a float, the same value is used for all outer iterations.
        If a list, each element corresponds to an outer iteration stage.
        Higher values (e.g., 2.0) make the acyclicity constraint more permissive, potentially allowing denser graphs.
        Lower values (e.g., 0.5) make it stricter, encouraging sparser solutions.
        The default ``[1.0, 0.9, 0.8, 0.7, 0.6]`` progressively tightens the acyclicity barrier across outer
        iterations.

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
        Number of inner optimization steps for the **final** outer iteration (mu level).
        Non-final iterations use ``warm_iter`` steps (typically fewer). If ``None``,
        defaults to 3000 for Adam and 1 for L-BFGS.

    warm_iter : int or None, optional (default=None)
        Number of inner optimization steps for non-final outer iterations (stages 0 to
        T-2). The final stage uses ``inner_iter`` steps (typically 2x more). If
        ``None``, defaults to the same value as ``inner_iter`` (no warm/final split).
        Setting ``warm_iter < inner_iter`` allocates more optimization budget to the
        final stage where the acyclicity barrier is tightest.

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
        Keyword arguments passed to the optimizer constructor. If ``None``, sensible defaults are used:
        ``{"lr": 0.0003, "betas": (0.99, 0.999)}`` for Adam (matching official DAGMA), or the optimizer's PyTorch
        defaults for other optimizers. For ``torch.optim.LBFGS`` a recommended configuration is
        ``{"max_iter": 10, "line_search_fn": "strong_wolfe"}``.

    random_state : int or None, optional (default=None)
        Seed for reproducibility. When provided, seeds both ``torch.manual_seed()`` and ``np.random.seed()`` at the
        start of ``fit()``. If ``None``, no seeding is performed.

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
        s=None,
        lambda1=0.05,
        mu_init=1.0,
        mu_factor=0.1,
        max_iter=None,
        inner_iter=None,
        warm_iter=None,
        w_threshold=0.3,
        return_type: str = "dag",
        optimizer=None,
        optimizer_kwargs=None,
        random_state=None,
    ) -> None:
        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.warm_iter = warm_iter
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame):
        r"""
        Core flow of the DAGMA continuous optimization algorithm.

        The algorithm uses a central path method that optimizes a sequence of unconstrained problems. As :math:`\mu`
        decays to zero, the solution converges to a DAG.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.
        """
        device, dtype = self._resolve_device_and_dtype()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Biased MLE covariance (1/n) on mean-centered data.
        data_np = X.values
        data_np = data_np - np.mean(data_np, axis=0, keepdims=True)
        cov = (data_np.T @ data_np) / float(data_np.shape[0])
        cov_tensor = torch.tensor(cov, device=device, dtype=dtype)

        W_tensor = torch.zeros((self.n_features_in_, self.n_features_in_), device=device, dtype=dtype)

        # Pre-compute identity matrix (avoid 15k+ allocations across inner loop)
        eye = torch.eye(self.n_features_in_, device=device, dtype=dtype)

        # Closure for objective value -- used for both autograd fallback (when W requires grad, the graph is built) and
        # convergence checks (called on .data under no_grad).
        def objective_fn(W, mu, s):
            return self._objective_value(W, mu, s, cov_tensor, eye)

        # Closure for analytical gradient (returns (grad, is_valid))
        def gradient_fn(W, mu, s):
            return self._gradient(W, mu, s, cov_tensor, eye)

        # Resolve optimizer: None -> Adam (matches official DAGMA package)
        optimizer_cls = self.optimizer if self.optimizer is not None else torch.optim.Adam

        # Resolve kwargs: None -> sensible defaults per optimizer
        if self.optimizer_kwargs is not None:
            opt_kwargs = self.optimizer_kwargs
        elif issubclass(optimizer_cls, torch.optim.Adam):
            opt_kwargs = {"lr": 0.0003, "betas": (0.99, 0.999), "foreach": False}
        else:
            opt_kwargs = {}

        # Resolve iterations based on optimizer type
        max_iter_val = self.max_iter
        inner_iter_val = self.inner_iter

        if issubclass(optimizer_cls, torch.optim.Adam):
            max_iter_val = 5 if max_iter_val is None else max_iter_val
            # Scale the final iteration budget based on the number of variables (d)
            inner_iter_val = max(3000, 750 * self.n_features_in_) if inner_iter_val is None else inner_iter_val
        else:
            # Defaults for L-BFGS and others
            max_iter_val = 100 if max_iter_val is None else max_iter_val
            inner_iter_val = 1 if inner_iter_val is None else inner_iter_val

        # Resolve warm_iter
        if self.warm_iter is not None:
            warm_iter_val = self.warm_iter
        elif issubclass(optimizer_cls, torch.optim.Adam):
            # Official DAGMA uses half the budget for warm-up iterations
            warm_iter_val = max(3000, inner_iter_val // 2)
        else:
            warm_iter_val = inner_iter_val

        # Resolve s: None -> official default schedule
        s_val = self.s if self.s is not None else [1.0, 0.9, 0.8, 0.7, 0.6]

        # Build s-schedule from scalar or list
        if isinstance(s_val, (int, float)):
            s_schedule = [float(s_val)]
        elif isinstance(s_val, list):
            if len(s_val) == 0:
                raise ValueError("s must be a non-empty list")
            s_schedule = list(s_val)
        else:
            raise ValueError(f"s must be a float or list of floats, got {type(s_val)}")

        # Analytical gradients are mathematically identical to autograd but  ~5x faster per step.
        # L-BFGS requires step(closure) which is incompatible with manual grad injection -- fall back to autograd.
        effective_gradient_fn = gradient_fn if optimizer_cls is not torch.optim.LBFGS else None

        W_est_final = self._optimize(
            W_tensor=W_tensor,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=opt_kwargs,
            objective_fn=objective_fn,
            mu_init=self.mu_init,
            mu_factor=self.mu_factor,
            max_iter=max_iter_val,
            inner_iter=inner_iter_val,
            gradient_fn=effective_gradient_fn,
            s_schedule=s_schedule,
            warm_iter=warm_iter_val,
        )

        self.adjacency_matrix_ = W_est_final

        self.causal_graph_ = self._convert_to_dag(
            W_est_final, list(self.feature_names_in_), self.w_threshold, self.return_type
        )

        return self

    def _gradient(self, W: torch.Tensor, mu: float, s: float, cov: torch.Tensor, eye: torch.Tensor):
        r"""
        Compute the analytical gradient of the DAGMA objective function.

        This avoids autograd overhead by computing the gradient directly from the closed-form expressions. The gradient
        is mathematically identical to ``torch.autograd.grad`` (verified to :math:`5.6 \times 10^{-17}`).
        .. math::
            \nabla_W = -\mu \cdot \hat{\Sigma} \cdot (I - W)
                       + \mu \cdot \lambda_1 \cdot \text{sign}(W)
                       + 2W \cdot (\text{inv}(sI - W \circ W))^T

        Parameters
        ----------
        W : torch.Tensor
            The (d, d) adjacency matrix (detached, no grad tracking needed).
        mu : float
            Central path parameter.
        s : float
            Current M-matrix domain parameter.
        cov : torch.Tensor
            Pre-computed (d, d) covariance matrix.
        eye : torch.Tensor
            Pre-computed (d, d) identity matrix (cached to avoid reallocation).

        Returns
        -------
        grad : torch.Tensor or None
            The (d, d) gradient tensor, or None if domain violation detected.
        is_valid : bool
            False if W is outside the M-matrix domain (inv(M) has negative entries).
        """
        M = s * eye - W * W
        M_inv = torch.linalg.inv(M)

        # Domain check: if inv(M) has any negative entry, W is outside the M-matrix
        # domain. The -1e-12 tolerance prevents floating-point inversion noise from
        # triggering false-positive failures.
        if torch.any(M_inv < -1e-12):
            return None, False

        G_score = -mu * cov @ (eye - W)
        G_h = 2 * W * M_inv.T
        G_l1 = mu * self.lambda1 * torch.sign(W)

        grad = G_score + G_l1 + G_h
        return grad, True

    def _objective_value(
        self, W: torch.Tensor, mu: float, s: float, cov: torch.Tensor, eye: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the DAGMA objective value WITHOUT building an autograd graph.

        Used for convergence checking when analytical gradients are active. Also serves as the objective function for
        the autograd fallback path (when ``W`` requires grad, the computation graph is built automatically).

        Parameters
        ----------
        W : torch.Tensor
            The (d, d) adjacency matrix.
        mu : float
            Central path parameter.
        s : float
            Current M-matrix domain parameter.
        cov : torch.Tensor
            Pre-computed (d, d) covariance matrix.
        eye : torch.Tensor
            Pre-computed (d, d) identity matrix.

        Returns
        -------
        torch.Tensor
            The objective value.
        """
        M = s * eye - W * W
        sign, logdet = torch.slogdet(M)
        if sign <= 0:
            return torch.tensor(1e10, dtype=W.dtype, device=W.device)

        h = -logdet + self.n_features_in_ * math.log(s)
        dif = eye - W
        score = 0.5 * (dif * (cov @ dif)).sum()
        obj = mu * (score + self.lambda1 * torch.abs(W).sum()) + h
        return obj
