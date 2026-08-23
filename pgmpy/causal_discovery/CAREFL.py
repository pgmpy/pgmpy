from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.model_selection import train_test_split

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.global_vars import config

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


@dataclass(frozen=True)
class FlowConfig:
    """Architecture of one fixed-order bivariate flow model."""

    num_flows: int
    hidden_dim: int
    hidden_layers: int


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings shared by both direction models."""

    batch_size: int
    max_epochs: int
    optimizer: str
    optimizer_kwargs: dict[str, Any]
    scheduler_kwargs: dict[str, Any]
    seed: int | None


_OPTIMIZER_VALID_KWARGS = {
    "adam": {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"},
    "sgd": {
        "lr",
        "momentum",
        "dampening",
        "weight_decay",
        "nesterov",
        "maximize",
    },
    "adamw": {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"},
}


def _validate_optimizer(
    optimizer: str,
    optimizer_kwargs: dict[str, Any],
) -> None:
    """Validate the optimizer name and keyword arguments."""
    if not isinstance(optimizer, str):
        raise ValueError(f"optimizer must be a string, got {type(optimizer)}.")

    name = optimizer.lower()
    if name not in _OPTIMIZER_VALID_KWARGS:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Supported optimizers are: {list(_OPTIMIZER_VALID_KWARGS)}.")
    if "params" in optimizer_kwargs:
        raise ValueError("'params' cannot be passed in optimizer_kwargs; CAREFL manages model parameters internally.")

    unknown = set(optimizer_kwargs) - _OPTIMIZER_VALID_KWARGS[name]
    if unknown:
        raise ValueError(f"Unknown optimizer_kwargs for '{optimizer}': {sorted(unknown)}.")


class _ConditionerMLP(nn.Module):
    """Private scalar-to-scalar MLP used for either scale or shift.

    This small helper avoids duplicating network construction in every flow
    layer. Its activation is an implementation detail and is not public API.
    """

    def __init__(self, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.LeakyReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Map a ``(batch, 1)`` parent tensor to a ``(batch, 1)`` value."""
        return self.network(x)


class _AffineARFlow(nn.Module):
    """One bivariate affine autoregressive layer with fixed ordering."""

    def __init__(self, hidden_dim: int, hidden_layers: int):
        super().__init__()
        self.s1 = nn.Parameter(torch.zeros(()))
        self.t1 = nn.Parameter(torch.zeros(()))
        self.scale_net = _ConditionerMLP(hidden_dim, hidden_layers)
        self.shift_net = _ConditionerMLP(hidden_dim, hidden_layers)

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        """Transform latent ``z`` to observations, preserving column order."""
        z1, z2 = z[:, 0:1], z[:, 1:2]
        x1 = torch.exp(self.s1) * z1 + self.t1
        x2 = torch.exp(self.scale_net(x1)) * z2 + self.shift_net(x1)
        return torch.cat([x1, x2], dim=1)

    def inverse(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return ``(z, log_abs_det_dx)`` using the analytical inverse.

        ``log_abs_det_dx`` is the per-sample log absolute determinant of
        ``dz / dx`` and has shape ``(batch,)``. No full Jacobian is built.
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        s2 = self.scale_net(x1)
        t2 = self.shift_net(x1)
        z1 = torch.exp(-self.s1) * (x1 - self.t1)
        z2 = torch.exp(-s2) * (x2 - t2)
        log_abs_det_dx = -self.s1 - s2.squeeze(1)
        return torch.cat([z1, z2], dim=1), log_abs_det_dx


class _CAREFLModel(nn.Module):
    """Private CAREFL density model for exactly one assumed causal ordering."""

    def __init__(self, flow_cfg: FlowConfig, train_cfg: TrainingConfig):
        super().__init__()
        self.flow_cfg = flow_cfg
        self.train_cfg = train_cfg
        self.flows = nn.ModuleList(
            [_AffineARFlow(flow_cfg.hidden_dim, flow_cfg.hidden_layers) for _ in range(flow_cfg.num_flows)]
        )
        self.register_buffer("base_loc", torch.tensor(0.0))
        self.register_buffer("base_scale", torch.tensor(1.0))

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        """Apply all flow layers in generative order, mapping ``z`` to ``x``."""
        x = z
        for flow in self.flows:
            x = flow(x)
        return x

    def inverse(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Invert all layers and sum per-sample inverse log-Jacobians."""
        z = x
        total_log_det = x.new_zeros(x.shape[0])
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def log_prob(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return one exact joint log-likelihood per row of ordered ``x``."""
        z, log_det = self.inverse(x)
        base_dist = torch.distributions.Laplace(self.base_loc, self.base_scale)
        return base_dist.log_prob(z).sum(dim=1) + log_det

    def fit_network(self, x_train: "torch.Tensor") -> "_CAREFLModel":
        """Minimize mini-batch negative mean log-likelihood and return ``self``.

        The optimizer and ``ReduceLROnPlateau`` scheduler are constructed from
        ``train_cfg``. Training state remains private; direction comparison is
        deliberately not performed here.
        """
        if self.train_cfg.seed is not None:
            torch.manual_seed(self.train_cfg.seed)

        self.to(device=x_train.device, dtype=x_train.dtype)
        _validate_optimizer(
            self.train_cfg.optimizer,
            self.train_cfg.optimizer_kwargs,
        )
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }[self.train_cfg.optimizer.lower()]
        optimizer = optimizer_cls(
            self.parameters(),
            **self.train_cfg.optimizer_kwargs,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **self.train_cfg.scheduler_kwargs,
        )

        self.train()
        num_samples = x_train.shape[0]
        for _ in range(self.train_cfg.max_epochs):
            permutation = torch.randperm(num_samples, device=x_train.device)
            epoch_loss = 0.0

            for start in range(0, num_samples, self.train_cfg.batch_size):
                batch = x_train[permutation[start : start + self.train_cfg.batch_size]]
                loss = -self.log_prob(batch).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]

            scheduler.step(epoch_loss / num_samples)

        return self


class CAREFL(BaseCausalDiscovery):
    """Discover causal direction between exactly two numerical variables.

    Two fixed-order affine autoregressive flow models are trained on the same
    train split. Their mean log-likelihoods are evaluated on the same held-out
    split, and the direction with the larger value is selected.

    Parameters
    ----------
    num_flows : int, default=2
        Number of stacked affine AR layers. Paper default for the basic
        synthetic causal-discovery experiment.
    hidden_dim : int, default=10
        Hidden units in each scale and shift conditioner. Paper default for the
        basic synthetic experiment.
    hidden_layers : int, default=1
        Hidden layers in each conditioner. Paper default for the basic
        synthetic experiment.
    batch_size : int, default=128
        Mini-batch size. Paper default for the basic synthetic experiment.
    max_epochs : int, default=200
        Number of training epochs per direction. Paper default for the basic
        synthetic experiment.
    test_size : float, default=0.2
        Fraction held out for direction comparison. This is a pgmpy API choice:
        the paper explicitly reports an 80/20 split for its cause-effect-pair
        experiments, but does not state it as a universal algorithm default.
    optimizer : str, default="adam"
        Optimizer name passed to the private CAREFL model.
    optimizer_kwargs : dict or None, default=None
        Keyword arguments passed to the private model's optimizer resolver.
    scheduler_kwargs : dict or None, default=None
        Keyword arguments passed to the private model's scheduler resolver.
    seed : int or None, default=42
        Seed for splitting, initialization, and mini-batch ordering. This is a
        pgmpy reproducibility choice, not a paper default.

    Notes
    -----
    A scaler and TensorBoard path are intentionally omitted from the first API.
    Scaling changes a density unless its Jacobian is accounted for, and neither
    option is required by CAREFL. Users may preprocess both columns before fit.

    Fitted attributes
    -----------------
    n_features_in_ : int
        Always 2 after successful fitting; set by ``BaseCausalDiscovery``.
    feature_names_in_ : numpy.ndarray
        The two input column labels in their original order, following sklearn.
    flow_config_ : FlowConfig
        Validated architecture configuration.
    train_config_ : TrainingConfig
        Training configuration passed to each private direction model.
    models_ : dict[tuple[object, object], _CAREFLModel]
        Maps ``(cause, effect)`` hypotheses to their fitted fixed-order models.
    log_likelihoods_ : dict[tuple[object, object], float]
        Mean held-out joint log-likelihood for each directional hypothesis.
    causal_score_ : float
        ``L(X -> Y) - L(Y -> X)`` for columns ``[X, Y]``: the empirical
        log-likelihood-ratio statistic from Equation (7), not a probability.
    causal_direction_ : tuple[object, object] or None
        ``(cause, effect)`` for the selected edge. ``None`` only for an exact
        likelihood tie, which the paper's sign rule leaves inconclusive.
    adjacency_matrix_ : pandas.DataFrame
        A labeled 2-by-2 binary adjacency matrix; rows are sources and columns
        are destinations. It has one edge, or no edge for an exact tie.
    causal_graph_ : pgmpy.base.DAG
        DAG containing both input nodes and the selected directed edge.
    """

    def __init__(
        self,
        num_flows: int = 2,
        hidden_dim: int = 10,
        hidden_layers: int = 1,
        batch_size: int = 128,
        max_epochs: int = 200,
        test_size: float = 0.2,
        optimizer: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        seed: int | None = 42,
    ):
        """Initialize the bivariate CAREFL estimator."""
        _check_soft_dependencies(
            "torch",
            msg="CAREFL requires PyTorch. Install it with: pip install torch",
        )
        super().__init__()
        self.num_flows = num_flows
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.test_size = test_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.seed = seed

    def _fit(self, X: pd.DataFrame):
        """Fit both causal orderings and construct the selected bivariate DAG."""
        # Validate CAREFL's bivariate data and parameters.
        if X.shape[1] != 2:
            raise ValueError(f"CAREFL requires exactly two variables, got {X.shape[1]}.")

        invalid_columns = [
            column
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(X[column]) or pd.api.types.is_bool_dtype(X[column])
        ]
        if invalid_columns:
            raise ValueError(f"CAREFL requires continuous numeric variables. Invalid columns: {invalid_columns}.")

        constant_columns = [column for column in X.columns if X[column].nunique(dropna=False) <= 1]
        if constant_columns:
            raise ValueError(f"CAREFL requires non-constant variables. Constant columns: {constant_columns}.")

        for name in (
            "num_flows",
            "hidden_dim",
            "hidden_layers",
            "batch_size",
            "max_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an integer greater than or equal to 1, got {value!r}.")

        if isinstance(self.test_size, bool) or not isinstance(self.test_size, Real) or not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be a number strictly between 0 and 1, got {self.test_size!r}.")

        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, Integral)):
            raise ValueError(f"seed must be an integer or None, got {self.seed!r}.")

        _validate_optimizer(self.optimizer, self.optimizer_kwargs or {})

        # Build architecture and training configs.
        self.flow_config_ = FlowConfig(
            num_flows=self.num_flows,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        optimizer = self.optimizer.lower()
        if optimizer == "adam":
            optimizer_kwargs = {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                **(self.optimizer_kwargs or {}),
            }
        else:
            optimizer_kwargs = dict(self.optimizer_kwargs or {})
        scheduler_kwargs = {
            "factor": 0.1,
            **(self.scheduler_kwargs or {}),
        }
        self.train_config_ = TrainingConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            seed=self.seed,
        )

        # Split once so both directions use the same rows.
        train_data, test_data = train_test_split(
            X,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
        )
        x, y = self.feature_names_in_

        # Train and score X -> Y and Y -> X.
        dtype = config.DTYPE if config.BACKEND == "torch" else torch.float32
        train_tensor = torch.tensor(
            np.ascontiguousarray(train_data[[x, y]].to_numpy()),
            dtype=dtype,
            device=config.DEVICE,
        )
        test_tensor = torch.tensor(
            np.ascontiguousarray(test_data[[x, y]].to_numpy()),
            dtype=dtype,
            device=config.DEVICE,
        )
        models = {}
        likelihoods = {}
        for cause, effect, ordered_train, ordered_test in (
            (x, y, train_tensor, test_tensor),
            (y, x, train_tensor[:, [1, 0]], test_tensor[:, [1, 0]]),
        ):
            model = _CAREFLModel(self.flow_config_, self.train_config_)
            model.fit_network(ordered_train)
            model.eval()
            with torch.no_grad():
                log_probs = model.log_prob(ordered_test)

            if log_probs.ndim != 1 or log_probs.shape[0] != ordered_test.shape[0]:
                raise ValueError("_CAREFLModel.log_prob must return one scalar per held-out observation.")
            if not bool(torch.isfinite(log_probs).all().item()):
                raise ValueError("_CAREFLModel.log_prob must return finite values.")

            models[(cause, effect)] = model
            likelihoods[(cause, effect)] = float(log_probs.mean().item())

        # Compare directions and build the graph.
        self.models_ = models
        self.log_likelihoods_ = likelihoods
        forward_ll = likelihoods[(x, y)]
        backward_ll = likelihoods[(y, x)]
        self.causal_score_ = forward_ll - backward_ll
        if forward_ll > backward_ll:
            self.causal_direction_ = (x, y)
        elif backward_ll > forward_ll:
            self.causal_direction_ = (y, x)
        else:
            self.causal_direction_ = None

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from([x, y])
        if self.causal_direction_ is not None:
            self.causal_graph_.add_edge(*self.causal_direction_)

        self.adjacency_matrix_ = pd.DataFrame(
            np.zeros((2, 2), dtype=int),
            index=[x, y],
            columns=[x, y],
        )
        if self.causal_direction_ is not None:
            self.adjacency_matrix_.loc[self.causal_direction_] = 1

        return self
