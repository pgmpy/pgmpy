from dataclasses import dataclass

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.causal_discovery._base import BaseCausalDiscovery

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


@dataclass
class NetworkConfig:
    hidden_dim: int
    scaler: object
    target_col: int | str | None


@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    optimizer: object
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

    def train(self, X_tensor):
        """Train the CASTLE model and return the adjacency matrix."""
        # TODO: Implement the CASTLE training loop.
        raise NotImplementedError("TBD")

    def get_W(self):
        """Compute the weighted adjacency matrix from input-layer weights."""
        return torch.stack(
            [(layer.weight * getattr(self, f"mask_{j}")).norm(dim=0) for j, layer in enumerate(self.input_layers)],
            dim=1,
        )


class CASTLE(BaseCausalDiscovery):
    """
    Supervised causal discovery using CASTLE.
    """

    def __init__(
        self,
        dag_weight: float = 1.0,
        sparsity_weight: float = 5.0,
        dag_penalty: float = 1.0,
        optimizer: object | None = None,
        batch_size: int = 32,
        hidden_dim: int = 32,
        edge_threshold: float = 0.3,
        target_col: int | str | None = None,
        max_epochs: int = 200,
        min_loss_improvement: float = 1e-4,
        early_stop_patience: int = 10,
        scaler: object | None = None,
        tensorboard_log_dir: str | None = None,
        seed: int = 42,
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

        self.network_config_ = NetworkConfig(
            hidden_dim=self.hidden_dim,
            scaler=self.scaler,
            target_col=self.target_col,
        )
        self.train_config_ = TrainingConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            optimizer=self.optimizer,
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
        self.causal_graph_ = None
        self.adjacency_matrix_ = None
        self.model_ = None
        self.scaler_ = None

        # TODO:
        # 1. Preprocess X: reorder columns so target is at index 0, fit StandardScaler,
        #    scale X, store scaler as self.scaler_ and column order as self.cols_
        # 2. Convert scaled data to a torch.Tensor
        # 3. Instantiate self.model_ = _CASTLEModel(num_inputs, network_config_,
        #    train_config_, reg_config_)
        # 4. Call W_final = self.model_.train(X_tensor)
        # 5. Build self.adjacency_matrix_ (pd.DataFrame from W_final, columns=self.cols_)
        # 6. Build self.causal_graph_ (pgmpy.base.DAG) from adjacency_matrix_
        # 7. Return self
