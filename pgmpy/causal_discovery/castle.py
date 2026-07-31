import itertools
from dataclasses import dataclass

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.preprocessing import StandardScaler

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.global_vars import config

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


@dataclass
class NetworkConfig:
    hidden_dim: int
    scaler: object
    target_col: str | None


@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    optimizer: str
    optimizer_kwargs: dict
    seed: int | None
    min_loss_improvement: float
    early_stop_patience: int
    tensorboard_log_dir: str | None


@dataclass
class RegularizationConfig:
    dag_weight: float
    sparsity_weight: float
    dag_penalty: float
    edge_threshold: float


_OPTIMIZER_VALID_KWARGS = {
    "adam": {"lr", "betas", "weight_decay", "amsgrad", "maximize", "eps"},
    "sgd": {"lr", "weight_decay", "momentum", "dampening", "nesterov", "maximize"},
    "adamw": {"lr", "betas", "weight_decay", "amsgrad", "maximize", "eps"},
}


def _validate_optimizer(optimizer: str, optimizer_kwargs: dict) -> None:
    """Validate optimizer name and kwargs, raising ValueError on any invalid input."""
    if not isinstance(optimizer, str):
        raise ValueError(f"optimizer must be a string, got {type(optimizer)}")
    name = optimizer.lower()
    if name not in _OPTIMIZER_VALID_KWARGS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Supported optimizers are: {list(_OPTIMIZER_VALID_KWARGS.keys())}."
        )
    if "params" in optimizer_kwargs:
        raise ValueError("'params' cannot be passed as an optimizer kwarg. CASTLE manages model parameters internally.")
    valid_keys = _OPTIMIZER_VALID_KWARGS[name]
    unknown = set(optimizer_kwargs) - valid_keys
    if unknown:
        unknown_str = ", ".join(f"'{k}'" for k in sorted(unknown))
        valid_str = ", ".join(f"'{k}'" for k in sorted(valid_keys))
        raise ValueError(
            f"Unknown optimizer_kwargs for '{optimizer}': {unknown_str}. Accepted kwargs are: {valid_str}."
        )


def _dag_constraint(W: "torch.Tensor") -> "torch.Tensor":
    """Compute the acyclicity constraint h(W) = tr(exp(W * W)) - d."""
    return torch.trace(torch.linalg.matrix_exp(W * W)) - W.shape[0]


class _CASTLEModel(nn.Module):
    """
    Internal masked autoencoder network used by CASTLE.
    """

    def __init__(
        self,
        num_inputs: int,
        network_cfg: NetworkConfig,
        train_cfg: TrainingConfig,
        reg_cfg: RegularizationConfig,
    ):
        """Initialize the internal CASTLE network."""
        super().__init__()
        self.num_inputs = num_inputs
        self.network_cfg = network_cfg
        self.train_cfg = train_cfg
        self.reg_cfg = reg_cfg

        if train_cfg.seed is not None:
            torch.manual_seed(train_cfg.seed)

        self.input_layers = nn.ModuleList(
            [nn.Linear(num_inputs, self.network_cfg.hidden_dim) for _ in range(num_inputs)]
        )
        for k in range(num_inputs):
            mask = torch.ones(self.network_cfg.hidden_dim, num_inputs)
            mask[:, k] = 0.0
            self.register_buffer(f"mask_{k}", mask)

        self.hidden_layers = nn.ModuleList([nn.Linear(self.network_cfg.hidden_dim, self.network_cfg.hidden_dim)])

        self.output_layers = nn.ModuleList([nn.Linear(self.network_cfg.hidden_dim, 1) for _ in range(num_inputs)])

    def forward(self, X):
        """Run a forward pass through the CASTLE network."""
        outputs = []
        for k in range(self.num_inputs):
            mask_k = getattr(self, f"mask_{k}")
            h = torch.relu(nn.functional.linear(X, self.input_layers[k].weight * mask_k, self.input_layers[k].bias))
            for layer in self.hidden_layers:
                h = torch.relu(layer(h))
            outputs.append(self.output_layers[k](h))

        Out = torch.cat(outputs, dim=1)
        out_0 = Out[:, 0:1]
        return Out, out_0

    def fit_network(self, X_tensor):
        """Train the CASTLE model and return the adjacency matrix."""
        if self.train_cfg.seed is not None:
            torch.manual_seed(self.train_cfg.seed)

        optimizer_cls = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }[self.train_cfg.optimizer.lower()]
        optimizer = optimizer_cls(self.parameters(), **(self.train_cfg.optimizer_kwargs or {}))

        best_loss = float("inf")
        patience_counter = 0
        N = X_tensor.shape[0]

        for _ in range(self.train_cfg.max_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, N, self.train_cfg.batch_size):
                X_batch = X_tensor[perm[start : start + self.train_cfg.batch_size]]

                Out, out_0 = self(X_batch)

                supervised_loss = nn.functional.mse_loss(out_0, X_batch[:, 0:1])
                reconstruction_loss = nn.functional.mse_loss(Out, X_batch)

                W = self.get_W()
                acyclicity_penalty = self.reg_cfg.dag_penalty * _dag_constraint(W) ** 2
                sparsity_loss = sum(
                    (layer.weight * getattr(self, f"mask_{k}")).abs().sum() for k, layer in enumerate(self.input_layers)
                )

                total_loss = supervised_loss + self.reg_cfg.dag_weight * (
                    reconstruction_loss + acyclicity_penalty + self.reg_cfg.sparsity_weight * sparsity_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            epoch_loss /= num_batches

            if best_loss - epoch_loss >= self.train_cfg.min_loss_improvement:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.train_cfg.early_stop_patience:
                    break

    def get_W(self):
        """Compute the weighted adjacency matrix from input-layer weights."""
        return torch.stack(
            [(layer.weight * getattr(self, f"mask_{j}")).norm(dim=0) for j, layer in enumerate(self.input_layers)],
            dim=1,
        )


class CASTLE(BaseCausalDiscovery):
    """
    Supervised causal discovery using CASTLE (CAusal STructure LEarning).

    Jointly trains a neural network predictor and learns a causal DAG as an
    auxiliary task via a masked autoencoder regularizer.

    Parameters
    ----------
    dag_weight : float, default 1.0
        Weight :math:`\\lambda` on the entire DAG regularization term
        (reconstruction loss + acyclicity penalty + sparsity penalty).
    sparsity_weight : float, default 5.0
        Weight :math:`\beta` on the group-lasso sparsity term applied to the
        input-layer weights. Controls edge sparsity in the learned DAG. This is
        independent of any ``weight_decay`` passed via ``optimizer_kwargs``.
    dag_penalty : float, default 1.0
        Initial augmented Lagrangian penalty coefficient :math:`\rho` for the
        acyclicity constraint. Doubled automatically when the constraint does
        not decrease sufficiently between epochs.
    optimizer : str, default 'adam'
        Name of the optimizer to use. Case-insensitive. Supported values:

        - ``'adam'``  -- :class:`torch.optim.Adam`
          accepted kwargs: ``lr``, ``betas``, ``eps``, ``weight_decay``,
          ``amsgrad``, ``maximize``
        - ``'sgd'``   -- :class:`torch.optim.SGD`
          accepted kwargs: ``lr``, ``weight_decay``, ``momentum``,
          ``dampening``, ``nesterov``, ``maximize``
        - ``'adamw'`` -- :class:`torch.optim.AdamW`
          accepted kwargs: ``lr``, ``betas``, ``eps``, ``weight_decay``,
          ``amsgrad``, ``maximize``

        Example usage::

            CASTLE(optimizer="adam", lr=1e-3)
            CASTLE(optimizer="sgd", lr=0.01, momentum=0.9)
            CASTLE(optimizer="adamw", lr=5e-4, weight_decay=1e-4)

    **optimizer_kwargs
        Additional keyword arguments forwarded to the optimizer constructor.
        ``params`` cannot be passed here -- CASTLE always manages model
        parameters internally.
    batch_size : int, default 32
        Mini-batch size used during training.
    hidden_dim : int, default 32
        Width :math:`h` of each sub-network's hidden layers.
    edge_threshold : float, default 0.3
        Edges with weight below this value are zeroed out in the final DAG.
    target_col : str or None, default None
        Column name to treat as the supervised target. If ``None``, defaults
        to the first column.
    max_epochs : int, default 200
        Maximum number of training epochs.
    min_loss_improvement : float, default 1e-4
        Minimum improvement in loss required to reset the early-stopping counter.
    early_stop_patience : int, default 10
        Number of epochs with no sufficient improvement before stopping early.
    scaler : object or None, default None
        Scaler used to standardize features. If ``None``, a
        :class:`sklearn.preprocessing.StandardScaler` is used.
    tensorboard_log_dir : str or None, default None
        Directory for TensorBoard logs. If ``None``, logging is disabled.
    seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dag_weight: float = 1.0,
        sparsity_weight: float = 5.0,
        dag_penalty: float = 1.0,
        optimizer: str = "adam",
        batch_size: int = 32,
        hidden_dim: int = 32,
        edge_threshold: float = 0.3,
        target_col: str | None = None,
        max_epochs: int = 200,
        min_loss_improvement: float = 1e-4,
        early_stop_patience: int = 10,
        scaler: object | None = None,
        tensorboard_log_dir: str | None = None,
        seed: int = 42,
        optimizer_kwargs: dict | None = None,
    ):
        """Initialize the CASTLE estimator."""
        _check_soft_dependencies(
            "torch",
            msg="CASTLE requires PyTorch. Install it with: pip install torch",
        )
        super().__init__()

        self.dag_weight = dag_weight
        self.sparsity_weight = sparsity_weight
        self.dag_penalty = dag_penalty
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.edge_threshold = edge_threshold
        self.target_col = target_col
        self.max_epochs = max_epochs
        self.min_loss_improvement = min_loss_improvement
        self.early_stop_patience = early_stop_patience
        self.scaler = scaler
        self.tensorboard_log_dir = tensorboard_log_dir
        self.seed = seed

    def _fit(self, X: pd.DataFrame):
        """Fit the CASTLE model and construct the causal DAG."""
        # Step 0: Validate inputs and preprocess data
        _validate_optimizer(self.optimizer, self.optimizer_kwargs or {})

        if X.shape[1] == 1:
            raise ValueError("CASTLE requires at least 2 columns (one target and at least one feature).")

        if self.target_col is None:
            target_col = X.columns[0]
        elif isinstance(self.target_col, str):
            if self.target_col not in X.columns:
                raise ValueError(f"target_col '{self.target_col}' not found in DataFrame columns {list(X.columns)}.")
            target_col = self.target_col
        else:
            raise ValueError(f"target_col must be a string column name or None, got {type(self.target_col).__name__}.")

        ordered_cols = [target_col] + [c for c in X.columns if c != target_col]
        self.feature_names_in_ = pd.Index(ordered_cols)
        self.predictor_names_ = ordered_cols[1:]
        X = X[ordered_cols]

        self.scaler_ = self.scaler if self.scaler is not None else StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Step 1: Build configuration objects
        self.network_config_ = NetworkConfig(
            hidden_dim=self.hidden_dim,
            scaler=self.scaler,
            target_col=self.target_col,
        )
        self.train_config_ = TrainingConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            seed=self.seed,
            min_loss_improvement=self.min_loss_improvement,
            early_stop_patience=self.early_stop_patience,
            tensorboard_log_dir=self.tensorboard_log_dir,
        )
        self.reg_config_ = RegularizationConfig(
            dag_weight=self.dag_weight,
            sparsity_weight=self.sparsity_weight,
            dag_penalty=self.dag_penalty,
            edge_threshold=self.edge_threshold,
        )

        # Step 2: Create tensor and train the CASTLE network
        dtype = config.DTYPE if config.BACKEND == "torch" else torch.float32
        X_tensor = torch.tensor(X_scaled, dtype=dtype, device=config.DEVICE)

        self.model_ = _CASTLEModel(self.n_features_in_, self.network_config_, self.train_config_, self.reg_config_)
        self.model_.fit_network(X_tensor)

        # Step 3: Extract learned weights and build adjacency matrix and causal DAG
        self.model_.eval()
        W_final = self.model_.get_W().detach().clone()
        W_final[W_final < self.reg_config_.edge_threshold] = 0.0

        self.adjacency_matrix_ = pd.DataFrame(
            W_final.cpu().numpy(), index=self.feature_names_in_, columns=self.feature_names_in_
        )

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(self.feature_names_in_)
        for src, dst in itertools.permutations(self.feature_names_in_, 2):
            if self.adjacency_matrix_.loc[src, dst] > 0:
                self.causal_graph_.add_edge(src, dst)

        return self
