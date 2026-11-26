import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.global_vars import config

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


_OPTIMIZER_VALID_KWARGS = {
    "adam": {"lr", "betas", "weight_decay", "amsgrad", "maximize", "eps"},
    "sgd": {"lr", "weight_decay", "momentum", "dampening", "nesterov", "maximize"},
    "adamw": {"lr", "betas", "weight_decay", "amsgrad", "maximize", "eps"},
    "rmsprop": {"lr", "alpha", "eps", "weight_decay", "momentum", "centered", "maximize"},
}


def _validate_optimizer(optimizer: str, optimizer_params: dict) -> None:
    """Validate that `optimizer` is one of the supported names and `optimizer_params` are valid for it."""
    if not isinstance(optimizer, str):
        raise ValueError(f"optimizer must be a string, got {type(optimizer)}")

    if optimizer_params is not None and not isinstance(optimizer_params, dict):
        raise ValueError(f"optimizer_params must be a dictionary, got {type(optimizer_params)}")

    name = optimizer.lower()
    if name not in _OPTIMIZER_VALID_KWARGS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Supported optimizers are: {list(_OPTIMIZER_VALID_KWARGS.keys())}."
        )

    params = optimizer_params or {}
    if "params" in params:
        raise ValueError(
            "'params' cannot be passed as an optimizer param. GraNDAG manages model parameters internally."
        )

    valid_keys = _OPTIMIZER_VALID_KWARGS[name]
    unknown = set(params) - valid_keys
    if unknown:
        unknown_str = ", ".join(f"'{k}'" for k in sorted(unknown))
        valid_str = ", ".join(f"'{k}'" for k in sorted(valid_keys))
        raise ValueError(
            f"Unknown optimizer_params for '{optimizer}': {unknown_str}. Accepted kwargs are: {valid_str}."
        )


def _dag_constraint(U: "torch.Tensor") -> "torch.Tensor":
    """Compute the acyclicity constraint h(U) = tr(exp(U)) - d."""
    return torch.trace(torch.linalg.matrix_exp(U)) - U.shape[1]


def _run_pns(X: np.ndarray, pns_threshold: float, seed: int, estimator=None) -> np.ndarray:
    """Preliminary Neighbourhood Selection via tree-based feature importances.

    Returns a boolean mask of shape ``(d, d)`` where ``True`` indicates a
    surviving parent candidate.
    """
    raise NotImplementedError


def _run_cam_pruning(X: np.ndarray, adj: np.ndarray, pruning_cutoff: float) -> np.ndarray:
    """CAM pruning: drop parents whose score p-value exceeds `pruning_cutoff`.

    Returns the pruned adjacency matrix.
    """
    raise NotImplementedError


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
        import copy

        import torch
        from torch import nn

        super().__init__()
        self.num_vars = num_vars
        self.network_cfg = network_cfg
        self.train_cfg = train_cfg
        self.reg_cfg = reg_cfg
        torch.manual_seed(train_cfg.seed)

        d = num_vars
        m = network_cfg.output_dim

        template = network_cfg.net
        if template is None:
            template = nn.Sequential(
                nn.Linear(d, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 1),
            )

        linears = []
        for mod in template.modules():
            has_params = any(True for _ in mod.parameters(recurse=False))
            if has_params:
                if not isinstance(mod, nn.Linear):
                    raise ValueError("All learnable layers must be nn.Linear.")
                linears.append(mod)

        if not linears:
            raise ValueError("All learnable layers must be nn.Linear.")
        if linears[0].in_features != d:
            raise ValueError(f"The first Linear layer must accept {d} inputs.")
        if linears[-1].out_features != m:
            raise ValueError(f"The last Linear layer must produce {m} outputs.")

        self.subnets = nn.ModuleList([copy.deepcopy(template) for _ in range(d)])
        self.log_var = nn.Parameter(torch.zeros(d))
        for subnet in self.subnets:
            for mod in subnet.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

        self._linears = []
        for subnet in self.subnets:
            subnet_linears = [mod for mod in subnet.modules() if isinstance(mod, nn.Linear)]
            self._linears.append(subnet_linears)

        mask = torch.ones(d, d) - torch.eye(d)
        self.register_buffer("adjacency", mask)
        self.register_buffer("_offdiag", mask.clone())

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        """Apply self-masking per variable and run each sub-network.

        Returns distribution parameters of shape ``(N, d, output_dim)``.
        """
        import torch

        N, d = X.shape
        outputs = []

        for j in range(d):
            mask_j = self.adjacency[:, j]
            x_masked = X * mask_j
            theta_j = self.subnets[j](x_masked)
            outputs.append(theta_j)

        theta = torch.stack(outputs, dim=1)
        return theta

    def _compute_log_likelihood(self, X: "torch.Tensor", theta: "torch.Tensor") -> "torch.Tensor":
        """Compute the configured or default Gaussian log-likelihood."""
        import math

        if self.network_cfg.log_likelihood is not None:
            return self.network_cfg.log_likelihood(X, theta)

        if theta.shape[-1] != 1:
            raise ValueError(
                "Default Gaussian log-likelihood requires output_dim=1 "
                f"(conditional mean); got output_dim={theta.shape[-1]}. Pass a custom "
                "log_likelihood for other output_dim values."
            )
        mu = theta[..., 0]
        log_var = self.log_var.clamp(-20.0, 20.0)
        var = log_var.exp()
        return -0.5 * math.log(2 * math.pi) - 0.5 * log_var - ((X - mu) ** 2) / (2 * var)

    def fit_network(self, X_tensor: "torch.Tensor", val_tensor: "torch.Tensor | None" = None) -> None:
        """Train the GraN-DAG model via augmented Lagrangian optimization.

        Runs the full training loop consisting of augmented Lagrangian subproblems.
        """
        import torch

        torch.manual_seed(self.train_cfg.seed)
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }[self.train_cfg.optimizer.lower()]

        self.lamb = self.reg_cfg.dag_multiplier_init
        self.mu = self.reg_cfg.dag_penalty
        h_prev = float("inf")
        h_val = float("inf")
        subproblem = 0
        N = X_tensor.shape[0]

        while h_val > self.reg_cfg.dag_constraint_tol and (
            self.train_cfg.max_subproblems is None or subproblem < self.train_cfg.max_subproblems
        ):
            optimizer_params = dict(self.train_cfg.optimizer_params or {})
            if "lr" not in optimizer_params:
                optimizer_params["lr"] = 1e-2 if subproblem == 0 else 1e-4
            optimizer = optimizer_cls(self.parameters(), **optimizer_params)

            best_val_loss = float("inf")
            patience_counter = 0

            for _ in range(self.train_cfg.max_epochs):
                self.train()
                permutation = torch.randperm(N, device=X_tensor.device)

                for start in range(0, N, self.train_cfg.batch_size):
                    X_batch = X_tensor[permutation[start : start + self.train_cfg.batch_size]]
                    theta = self(X_batch)
                    negative_log_likelihood = -self._compute_log_likelihood(X_batch, theta).sum(dim=1).mean()
                    A = self.get_A()
                    h = _dag_constraint(A)
                    loss = negative_log_likelihood + self.lamb * h + (self.mu / 2) * h**2

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        self.adjacency.mul_(self.get_A() >= self.reg_cfg.edge_threshold)

                if val_tensor is not None:
                    self.eval()
                    with torch.no_grad():
                        theta = self(val_tensor)
                        negative_log_likelihood = -self._compute_log_likelihood(val_tensor, theta).sum(dim=1).mean()
                        h = _dag_constraint(self.get_A())
                        val_loss = negative_log_likelihood + self.lamb * h + (self.mu / 2) * h**2
                        val_loss = val_loss.item()

                    if best_val_loss - val_loss >= self.train_cfg.min_loss_improvement:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.train_cfg.early_stop_patience:
                            break

            with torch.no_grad():
                h_val = _dag_constraint(self.get_A()).item()

            self.lamb = self.lamb + self.mu * h_val
            if h_val > self.reg_cfg.dag_penalty_growth_threshold * h_prev:
                self.mu = self.mu * self.reg_cfg.dag_penalty_growth_factor
            h_prev = h_val
            subproblem += 1

    def get_A(self) -> "torch.Tensor":
        """Compute weighted adjacency matrix from connectivity products."""
        import torch

        d = self.num_vars
        cols = []
        for j in range(d):
            Ws = self._linears[j]
            prod = Ws[0].weight.abs() * self.adjacency[:, j]
            for W in Ws[1:]:
                prod = W.weight.abs() @ prod
            cols.append(prod.sum(dim=0))

        A = torch.stack(cols, dim=1)
        return A * self._offdiag

    def get_jacobian(self, X: "torch.Tensor", chunk_size: int | None = None) -> "torch.Tensor":
        """Compute expected absolute Jacobian matrix over the dataset."""
        import torch
        from torch import autograd

        N, d = X.shape
        Jc = torch.zeros(d, d, device=X.device, dtype=X.dtype)

        if chunk_size is None:
            chunk_size = N

        # Detach model parameters to save memory during multiple backward passes
        original_req_grad = {p: p.requires_grad for p in self.parameters()}
        for p in self.parameters():
            p.requires_grad = False

        for i in range(0, N, chunk_size):
            X_chunk = X[i : i + chunk_size].detach().clone().requires_grad_(True)
            chunk_N = X_chunk.shape[0]

            theta = self.forward(X_chunk)
            logp = self._compute_log_likelihood(X_chunk, theta)

            ones = torch.ones(chunk_N, device=X_chunk.device, dtype=X_chunk.dtype)
            for j in range(d):
                g = autograd.grad(
                    logp[:, j],
                    X_chunk,
                    grad_outputs=ones,
                    retain_graph=(j < d - 1),
                )[0]
                Jc[j, :] += g.abs().sum(dim=0)

        # Restore original requires_grad state
        for p in self.parameters():
            p.requires_grad = original_req_grad[p]

        Jc = Jc / N
        J = Jc.t() * self.adjacency
        return J.detach()

    def _threshold_to_dag(self, J: "torch.Tensor") -> "torch.Tensor":
        """Threshold Jacobian entries and iteratively remove edges to form a DAG."""
        import networkx as nx
        import torch

        W = J.detach().clone().float()
        W.fill_diagonal_(0)
        W[W < self.reg_cfg.edge_threshold] = 0

        A = W > 0
        if nx.is_directed_acyclic_graph(nx.from_numpy_array(A.cpu().numpy(), create_using=nx.DiGraph)):
            return A.to(J.dtype)

        ts = torch.unique(W[W > 0])
        lo, hi = 0, len(ts) - 1
        EPS = 1e-8
        A = W > ts[hi] + EPS
        if not nx.is_directed_acyclic_graph(nx.from_numpy_array(A.cpu().numpy(), create_using=nx.DiGraph)):
            return torch.zeros_like(W).to(J.dtype)

        while lo < hi:
            mid = (lo + hi) // 2
            A = W > ts[mid] + EPS
            if nx.is_directed_acyclic_graph(nx.from_numpy_array(A.cpu().numpy(), create_using=nx.DiGraph)):
                hi = mid
            else:
                lo = mid + 1

        return (W > ts[lo] + EPS).to(J.dtype)


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
        Defaults to a built-in 2-layer MLP with Leaky-ReLU activations. All
        learnable layers must be ``nn.Linear``, with the first layer accepting
        ``d`` inputs and the last layer producing ``output_dim`` outputs.
    output_dim : int, default 1
        Number of output neurons per sub-network (distribution parameters).
    log_likelihood : callable or None, default None
        Per-sample log-probability function with signature
        ``fn(X, theta) -> Tensor`` batched over all variables at once.
        Defaults to Gaussian log-likelihood.
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
        Name of the optimizer to use. Case-insensitive. Supported values:

        - ``'rmsprop'`` -- :class:`torch.optim.RMSprop`
          accepted kwargs: ``lr``, ``alpha``, ``eps``, ``weight_decay``,
          ``momentum``, ``centered``, ``maximize``
        - ``'adam'``    -- :class:`torch.optim.Adam`
          accepted kwargs: ``lr``, ``betas``, ``eps``, ``weight_decay``,
          ``amsgrad``, ``maximize``
        - ``'sgd'``     -- :class:`torch.optim.SGD`
          accepted kwargs: ``lr``, ``weight_decay``, ``momentum``,
          ``dampening``, ``nesterov``, ``maximize``
        - ``'adamw'``   -- :class:`torch.optim.AdamW`
          accepted kwargs: ``lr``, ``betas``, ``eps``, ``weight_decay``,
          ``amsgrad``, ``maximize``
    optimizer_params : dict or None, default None
        Keyword arguments forwarded to the optimizer constructor. When no
        learning rate is supplied, uses ``1e-2`` for the first augmented-
        Lagrangian subproblem and ``1e-4`` thereafter. An explicitly supplied
        learning rate is used for every subproblem. Unknown kwargs for the
        chosen optimizer raise ``ValueError`` during validation.
    batch_size : int, default 64
        Mini-batch size for the inner optimization loop.
    val_size : float, default 0.1
        Fraction of data held out for early stopping within each subproblem.
    max_epochs : int, default 200
        Maximum number of training epochs per subproblem. Acts as the sole
        bound on the inner loop when ``val_size`` is ``0.0``, since early
        stopping is disabled in that case.
    min_loss_improvement : float, default 1e-4
        Minimum decrease in the validation augmented-Lagrangian objective to
        count as an improvement.
    early_stop_patience : int, default 5
        Epochs without improvement before stopping a subproblem early.
    tensorboard_log_dir : str or None, default None
        Directory for TensorBoard logs. If ``None``, logging is disabled.
    seed : int, default 42
        Random seed for reproducibility.
    edge_threshold : float, default 1e-4
        Connectivity values below this threshold are permanently masked during
        training. It is also used for final Jacobian thresholding.
    pns_threshold : float or None, default None
        If set, run Preliminary Neighbourhood Selection before training.
    pruning_cutoff : float or None, default None
        If set, apply CAM pruning after Jacobian thresholding.
    """

    def __init__(
        self,
        net=None,
        output_dim: int = 1,
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
        import torch

        # Step 0: Validate inputs.
        _validate_optimizer(self.optimizer, self.optimizer_params or {})

        if self.scaler is not None and (not hasattr(self.scaler, "fit") or not hasattr(self.scaler, "transform")):
            raise ValueError(
                "scaler must implement fit(X) and transform(X); "
                f"got {type(self.scaler)} which is missing one or both methods."
            )

        if X.shape[1] < 2:
            raise ValueError("GraNDAG requires at least 2 variables; got X with only 1 column.")

        if self.pns_threshold is not None:
            raise NotImplementedError("PNS is not yet implemented; pass pns_threshold=None.")

        if self.pruning_cutoff is not None:
            raise NotImplementedError("CAM pruning is not yet implemented; pass pruning_cutoff=None.")

        self.feature_names_in_ = pd.Index(X.columns)
        d = X.shape[1]

        # Step 1: Scale.
        self.scaler_ = self.scaler if self.scaler is not None else StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Step 2: Train/val split.
        if self.val_size and self.val_size > 0:
            X_train, X_val = train_test_split(X_scaled, test_size=self.val_size, random_state=self.seed)
        else:
            X_train, X_val = X_scaled, None

        # Step 3: Build tensors.
        dtype = config.DTYPE if config.BACKEND == "torch" else torch.float32
        X_train_tensor = torch.tensor(X_train, dtype=dtype, device=config.DEVICE)
        X_val_tensor = torch.tensor(X_val, dtype=dtype, device=config.DEVICE) if X_val is not None else None

        # Step 4: Build config dataclasses.
        self.network_config_ = GraNDAGNetworkConfig(
            net=self.net,
            output_dim=self.output_dim,
            log_likelihood=self.log_likelihood,
            scaler=self.scaler,
        )
        self.train_config_ = GraNDAGTrainingConfig(
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            batch_size=self.batch_size,
            val_size=self.val_size,
            max_epochs=self.max_epochs,
            min_loss_improvement=self.min_loss_improvement,
            early_stop_patience=self.early_stop_patience,
            max_subproblems=self.max_subproblems,
            tensorboard_log_dir=self.tensorboard_log_dir,
            seed=self.seed,
        )
        self.reg_config_ = GraNDAGRegularizationConfig(
            dag_multiplier_init=self.dag_multiplier_init,
            dag_penalty=self.dag_penalty,
            dag_penalty_growth_factor=self.dag_penalty_growth_factor,
            dag_penalty_growth_threshold=self.dag_penalty_growth_threshold,
            dag_constraint_tol=self.dag_constraint_tol,
            edge_threshold=self.edge_threshold,
        )

        # Step 5: Instantiate and train the internal model.
        self.model_ = _GraNDAGModel(d, self.network_config_, self.train_config_, self.reg_config_)
        self.model_.fit_network(X_train_tensor, X_val_tensor)

        # Step 6: Extract the final DAG from the expected absolute Jacobian.
        self.model_.eval()
        X_tensor = torch.tensor(X_scaled, dtype=dtype, device=config.DEVICE)
        J = self.model_.get_jacobian(X_tensor)
        adj = self.model_._threshold_to_dag(J)

        self.adjacency_matrix_ = pd.DataFrame(
            adj.detach().cpu().numpy(),
            index=self.feature_names_in_,
            columns=self.feature_names_in_,
        )

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(self.feature_names_in_)
        for src, dst in itertools.permutations(self.feature_names_in_, 2):
            if self.adjacency_matrix_.loc[src, dst] > 0:
                self.causal_graph_.add_edge(src, dst)

        return self
