from dataclasses import dataclass

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.causal_discovery._base import BaseCausalDiscovery

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


@dataclass
class GraNDAGNetworkConfig:
    net: "nn.Module | None"
    output_dim: int
    log_likelihood: object
    scaler: object


@dataclass
class GraNDAGTrainingConfig:
    optimizer: str
    optimizer_params: dict
    batch_size: int
    val_size: float
    max_epochs: int
    min_loss_improvement: float
    early_stop_patience: int
    max_subproblems: int | None
    tensorboard_log_dir: str | None
    seed: int


@dataclass
class GraNDAGRegularizationConfig:
    dag_multiplier_init: float
    dag_penalty: float
    dag_penalty_growth_factor: float
    dag_penalty_growth_threshold: float
    dag_constraint_tol: float
    edge_threshold: float


def _validate_optimizer(optimizer: str, optimizer_params: dict) -> None:
    """Validate that `optimizer` resolves to a torch.optim class and `optimizer_params` is a dict."""
    if not isinstance(optimizer, str):
        raise ValueError(f"optimizer must be a string, got {type(optimizer)}")

    if optimizer_params is not None and not isinstance(optimizer_params, dict):
        raise ValueError(f"optimizer_params must be a dictionary, got {type(optimizer_params)}")

    import torch

    valid_opts = [
        name
        for name in dir(torch.optim)
        if isinstance(getattr(torch.optim, name), type)
        and issubclass(getattr(torch.optim, name), torch.optim.Optimizer)
        and name != "Optimizer"
    ]
    valid_opts_lower = {name.lower(): name for name in valid_opts}

    if optimizer.lower() not in valid_opts_lower:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Supported optimizers are: {valid_opts}.")

    if optimizer_params is not None and "params" in optimizer_params:
        raise ValueError(
            "'params' cannot be passed as an optimizer param. GraNDAG manages model parameters internally."
        )


def _dag_constraint(U: "torch.Tensor") -> "torch.Tensor":
    """Compute the acyclicity constraint h(U) = tr(exp(U)) - d."""
    return torch.trace(torch.linalg.matrix_exp(U)) - U.shape[1]


def _run_pns(X: np.ndarray, pns_threshold: float, seed: int, estimator=None) -> np.ndarray:
    """Preliminary Neighbourhood Selection via tree-based feature importances.

    Returns a boolean mask of shape ``(d, d)`` where ``True`` indicates a
    surviving parent candidate.
    """
    # Step 1: If `estimator` is None, instantiate the default `sklearn.ensemble.ExtraTreesRegressor(random_state=seed)`.
    # Step 2: Initialize a boolean mask of shape (d, d) with False.
    # Step 3: Loop over each variable `j` from 0 to d-1.
    # Step 4: For variable `j`, prepare target y = X[:, j] and features X_rest = X without column `j`.
    # Step 5: Fit the estimator on X_rest and y.
    # Step 6: Verify the fitted estimator has a `feature_importances_` attribute; raise TypeError if not.
    # Step 7: Calculate the importance threshold for variable `j` (pns_threshold * mean(feature_importances_)).
    # Step 8: Set the mask for column `j` (excluding the diagonal) to True for features with importance >= threshold.
    # Step 9: Return the boolean mask.
    raise NotImplementedError


def _run_cam_pruning(X: np.ndarray, adj: np.ndarray, pruning_cutoff: float) -> np.ndarray:
    """CAM pruning: drop parents whose score p-value exceeds `pruning_cutoff`.

    Returns the pruned adjacency matrix.
    """
    # Step 1: Initialize a copy of the adjacency matrix to store the pruned graph.
    # Step 2: Loop over each node (variable) in the graph.
    # Step 3: For each node, identify its current parents from the adjacency matrix.
    # Step 4: Use pgmpy's CAM score implementation (or a fallback OLS implementation) to score each parent.
    # Step 5: If the p-value of a parent's score exceeds `pruning_cutoff`,
    # remove the edge (set to 0 in the pruned adj matrix).
    # Step 6: Return the pruned adjacency matrix.
    raise NotImplementedError


def _is_acyclic(A: "torch.Tensor") -> bool:
    """Helper to check if a binary adjacency tensor forms a DAG."""
    import networkx as nx

    return nx.is_directed_acyclic_graph(nx.from_numpy_array(A.cpu().numpy(), create_using=nx.DiGraph))


def _threshold_to_dag(J: "torch.Tensor", edge_threshold: float) -> "torch.Tensor":
    """Threshold Jacobian entries and iteratively remove edges to form a DAG.

    Follows the GraN-DAG post-training acyclicity step (Lachapelle et al.,
    ICLR 2020, §3.4 + Appendix A.2). Removes edges starting from the lowest
    Jacobian weight upward until the graph is completely acyclic.

    Parameters
    ----------
    J : torch.Tensor
        Expected absolute Jacobian matrix of shape (d, d).
    edge_threshold : float
        Entries in J below this value are zeroed before cycle removal.

    Returns
    -------
    torch.Tensor
        Binary (d, d) adjacency tensor with zero diagonal, guaranteed acyclic.
    """
    import torch

    W = J.detach().clone().float()
    W.fill_diagonal_(0)
    W[W < edge_threshold] = 0

    A = W > 0
    if _is_acyclic(A):
        return A.to(J.dtype)

    ts = torch.unique(W[W > 0])  # ascending
    lo, hi = 0, len(ts) - 1  # invariant: A(ts[hi]) is acyclic
    EPS = 1e-8
    if not _is_acyclic(W > ts[hi] + EPS):
        return torch.zeros_like(W).to(J.dtype)

    while lo < hi:
        mid = (lo + hi) // 2
        if _is_acyclic(W > ts[mid] + EPS):
            hi = mid
        else:
            lo = mid + 1

    return (W > ts[lo] + EPS).to(J.dtype)


class _GraNDAGModel(nn.Module):
    """PyTorch NN ensemble for per-variable conditional distribution learning and DAG structure recovery."""

    def __init__(
        self,
        num_vars: int,
        network_cfg: GraNDAGNetworkConfig,
        train_cfg: GraNDAGTrainingConfig,
        reg_cfg: GraNDAGRegularizationConfig,
    ):
        """Initialize the internal GraN-DAG network."""
        raise NotImplementedError

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        """Apply self-masking per variable and run each sub-network.

        Returns distribution parameters of shape ``(N, d, output_dim)``.
        """
        raise NotImplementedError

    def train(self, X_tensor=True, val_tensor=None):
        """Train the GraN-DAG model via augmented Lagrangian optimization.

        When called with a bool, delegates to ``nn.Module.train(mode)``.
        When called with a Tensor, runs the full training loop.
        """
        # Delegate to nn.Module.train(mode) when called with a bool.
        # Seed, then init lambda, mu and h_prev for the augmented Lagrangian.
        # Outer loop over subproblems: capped by max_subproblems, exit when h <= dag_constraint_tol.
        #   Inner loop over epochs (capped by max_epochs): minibatch NLL + lambda*h + (mu/2)*h**2.
        #   Early stop on validation NLL when val_size > 0; reset patience each subproblem.
        #   After each subproblem: update lambda/mu from h and h_prev.
        # Return the thresholded adjacency matrix.
        raise NotImplementedError

    def get_A(self) -> "torch.Tensor":
        """Compute weighted adjacency matrix from connectivity products."""
        raise NotImplementedError

    def get_jacobian(self, X: "torch.Tensor") -> "torch.Tensor":
        """Compute expected absolute Jacobian matrix over the dataset."""
        raise NotImplementedError

    def _update_lagrangian(self, h_val: float, h_prev: float) -> None:
        """Update Lagrangian coefficients after each subproblem."""
        raise NotImplementedError


class GraNDAG(BaseCausalDiscovery):
    """Causal discovery using GraN-DAG (Gradient-based Neural DAG Learning).

    Parameterizes each variable's conditional distribution with a neural
    network, learns the DAG structure via an augmented Lagrangian method with
    a continuous acyclicity constraint, and recovers the final graph by
    thresholding the expected absolute Jacobian matrix.

    Parameters
    ----------
    net : nn.Module or None, default None
        Neural network template, cloned ``d`` times via ``copy.deepcopy``.
        Defaults to a built-in 2-layer MLP with sigmoid activations. All
        learnable layers must be ``nn.Linear``, with the first layer accepting
        ``d`` inputs and the last layer producing ``output_dim`` outputs.
    output_dim : int, default 2
        Number of output neurons per sub-network (distribution parameters).
    log_likelihood : callable or None, default None
        Per-sample log-probability function with signature
        ``fn(x_j, theta) -> Tensor``. Defaults to Gaussian log-likelihood.
    scaler : object or None, default None
        Optional scaler for input normalization (e.g.,
        :class:`sklearn.preprocessing.StandardScaler`). Must implement
        ``fit(X)`` and ``transform(X)``.
    dag_multiplier_init : float, default 0.0
        Initial Lagrangian multiplier for the acyclicity constraint.
    dag_penalty : float, default 1e-3
        Initial quadratic penalty coefficient on the acyclicity violation.
    dag_penalty_growth_factor : float, default 10.0
        Multiplier applied to the penalty when the constraint has not
        decreased sufficiently.
    dag_penalty_growth_threshold : float, default 0.9
        Ratio threshold for triggering a penalty increase.
    dag_constraint_tol : float, default 1e-8
        Stop the outer loop when the acyclicity constraint is below this value.
    max_subproblems : int or None, default None
        Hard cap on augmented Lagrangian outer iterations. ``None`` means no
        cap.
    optimizer : str, default 'rmsprop'
        Name of a ``torch.optim`` optimizer class (case-insensitive).
    optimizer_params : dict or None, default None
        Keyword arguments forwarded to the optimizer constructor. Defaults to
        ``{"lr": 1e-3}`` when ``None``.
    batch_size : int, default 64
        Mini-batch size for the inner optimization loop.
    val_size : float, default 0.1
        Fraction of data held out for early stopping within each subproblem.
    max_epochs : int, default 200
        Maximum number of training epochs per subproblem. Acts as the sole
        bound on the inner loop when ``val_size`` is ``0.0``, since early
        stopping is disabled in that case.
    min_loss_improvement : float, default 1e-4
        Minimum decrease in validation NLL to count as an improvement.
    early_stop_patience : int, default 5
        Epochs without improvement before stopping a subproblem early.
    tensorboard_log_dir : str or None, default None
        Directory for TensorBoard logs. If ``None``, logging is disabled.
    seed : int, default 42
        Random seed for reproducibility.
    edge_threshold : float, default 1e-4
        Entries in the Jacobian matrix below this value are zeroed.
    pns_threshold : float or None, default None
        If set, run Preliminary Neighbourhood Selection before training.
    pruning_cutoff : float or None, default None
        If set, apply CAM pruning after Jacobian thresholding.
    """

    def __init__(
        self,
        net=None,
        output_dim: int = 2,
        log_likelihood=None,
        scaler=None,
        dag_multiplier_init: float = 0.0,
        dag_penalty: float = 1e-3,
        dag_penalty_growth_factor: float = 10.0,
        dag_penalty_growth_threshold: float = 0.9,
        dag_constraint_tol: float = 1e-8,
        max_subproblems: int | None = None,
        optimizer: str = "rmsprop",
        optimizer_params: dict | None = None,
        batch_size: int = 64,
        val_size: float = 0.1,
        max_epochs: int = 200,
        min_loss_improvement: float = 1e-4,
        early_stop_patience: int = 5,
        tensorboard_log_dir: str | None = None,
        seed: int = 42,
        edge_threshold: float = 1e-4,
        pns_threshold: float | None = None,
        pruning_cutoff: float | None = None,
    ):
        """Initialize the GraNDAG estimator."""
        _check_soft_dependencies(
            "torch",
            msg="GraNDAG requires PyTorch. Install it with: pip install torch",
        )
        super().__init__()

        self.net = net
        self.output_dim = output_dim
        self.log_likelihood = log_likelihood
        self.scaler = scaler
        self.dag_multiplier_init = dag_multiplier_init
        self.dag_penalty = dag_penalty
        self.dag_penalty_growth_factor = dag_penalty_growth_factor
        self.dag_penalty_growth_threshold = dag_penalty_growth_threshold
        self.dag_constraint_tol = dag_constraint_tol
        self.max_subproblems = max_subproblems
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.val_size = val_size
        self.max_epochs = max_epochs
        self.min_loss_improvement = min_loss_improvement
        self.early_stop_patience = early_stop_patience
        self.tensorboard_log_dir = tensorboard_log_dir
        self.seed = seed
        self.edge_threshold = edge_threshold
        self.pns_threshold = pns_threshold
        self.pruning_cutoff = pruning_cutoff

    def _fit(self, X: pd.DataFrame):
        """Fit the GraN-DAG model and construct the causal DAG."""
        # Step 0: Validate optimizer, net template and scaler; reject d < 2; set cols_.
        # Step 0b: Run PNS and set pns_mask_ when pns_threshold is not None.
        # Step 1: Build the network, training and regularization configs.
        # Step 2: Scale, split off val_size, move to tensors, train _GraNDAGModel.
        # Step 3: Jacobian -> threshold -> optional CAM pruning -> adjacency_matrix_, causal_graph_.
        raise NotImplementedError
