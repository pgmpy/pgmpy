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


class _ConditionerMLP(nn.Module):
    """Private scalar-to-scalar MLP used for either scale or shift.

    This small helper avoids duplicating network construction in every flow
    layer. Its activation is an implementation detail and is not public API.
    """

    def __init__(self, hidden_dim: int, hidden_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Map a ``(batch, 1)`` parent tensor to a ``(batch, 1)`` value."""
        raise NotImplementedError


class _AffineARFlow(nn.Module):
    """One bivariate affine autoregressive layer with fixed ordering."""

    def __init__(self, hidden_dim: int, hidden_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        """Transform latent ``z`` to observations, preserving column order."""
        raise NotImplementedError

    def inverse(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return ``(z, log_abs_det_dx)`` using the analytical inverse.

        ``log_abs_det_dx`` is the per-sample log absolute determinant of
        ``dz / dx`` and has shape ``(batch,)``. No full Jacobian is built.
        """
        raise NotImplementedError


class _CAREFLModel(nn.Module):
    """Private CAREFL density model for exactly one assumed causal ordering."""

    def __init__(self, flow_cfg: FlowConfig, train_cfg: TrainingConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        """Apply all flow layers in generative order, mapping ``z`` to ``x``."""
        raise NotImplementedError

    def inverse(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Invert all layers and sum per-sample inverse log-Jacobians."""
        raise NotImplementedError

    def log_prob(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return one exact joint log-likelihood per row of ordered ``x``."""
        raise NotImplementedError

    def fit_network(self, x_train: "torch.Tensor") -> "_CAREFLModel":
        """Minimize mini-batch negative mean log-likelihood and return ``self``.

        The optimizer and ``ReduceLROnPlateau`` scheduler are constructed from
        ``train_cfg``. Training state remains private; direction comparison is
        deliberately not performed here.
        """
        raise NotImplementedError


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

        # Build architecture and training configs.
        self.flow_config_ = FlowConfig(
            num_flows=self.num_flows,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        optimizer_kwargs = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            **(self.optimizer_kwargs or {}),
        }
        scheduler_kwargs = {
            "factor": 0.1,
            **(self.scheduler_kwargs or {}),
        }
        self.train_config_ = TrainingConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            optimizer=self.optimizer,
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
        models = {}
        likelihoods = {}
        for cause, effect in ((x, y), (y, x)):
            train_tensor = torch.tensor(
                train_data[[cause, effect]].to_numpy(),
                dtype=dtype,
                device=config.DEVICE,
            )
            test_tensor = torch.tensor(
                test_data[[cause, effect]].to_numpy(),
                dtype=dtype,
                device=config.DEVICE,
            )

            model = _CAREFLModel(self.flow_config_, self.train_config_)
            model.fit_network(train_tensor)
            model.eval()
            with torch.no_grad():
                log_probs = model.log_prob(test_tensor)

            if log_probs.ndim != 1 or log_probs.shape[0] != test_tensor.shape[0]:
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
