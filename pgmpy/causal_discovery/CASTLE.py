import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.preprocessing import StandardScaler

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery

torch = _safe_import("torch")
nn = _safe_import("torch.nn")
F = _safe_import("torch.nn.functional")


def _dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """Compute the DAG acyclicity constraint h(W).

    Uses the matrix exponential formulation:
        h(W) = trace(exp(W ⊙ W)) - d

    where d = W.shape[0] and ⊙ is elementwise (Hadamard) product.
    h(W) == 0 iff W encodes a DAG.

    Parameters
    ----------
    W : torch.Tensor of shape (d, d)
        Weighted adjacency matrix.

    Returns
    -------
    h : torch.Tensor (scalar)
        The acyclicity constraint value.
    """
    d = W.shape[0]
    return torch.trace(torch.linalg.matrix_exp(W * W)) - d


class _CASTLEModel(nn.Module):
    """Internal masked-autoencoder network for CASTLE.

    Each feature *i* has its own input layer whose *i*-th row is permanently
    masked to zero, preventing the sub-network for *i* from seeing its own
    value.  A single hidden layer is shared across all sub-networks.

    Parameters
    ----------
    num_inputs : int
        Number of input features (= number of DAG nodes).
    n_hidden : int
        Width of the hidden layers.
    """

    def __init__(self, num_inputs: int, n_hidden: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden

        self.input_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        for i in range(num_inputs):
            self.input_layers.append(nn.Linear(num_inputs, n_hidden, bias=True))

            # Mask: all 1s except row i is all 0s — prevents self-causation
            mask_i = torch.ones((num_inputs, n_hidden))
            mask_i[i, :] = 0.0
            self.register_buffer(f"mask_{i}", mask_i)

            self.output_layers.append(nn.Linear(n_hidden, 1, bias=True))

        # One shared hidden layer across all sub-networks
        self.hidden_layer = nn.Linear(n_hidden, n_hidden, bias=True)

    def forward(self, X: torch.Tensor):
        outs = []
        out_0 = None
        for i in range(self.num_inputs):
            mask_i = getattr(self, f"mask_{i}")
            # input_layers[i].weight shape: (n_hidden, num_inputs)
            # mask_i shape: (num_inputs, n_hidden) → mask_i.T: (n_hidden, num_inputs)
            weight_masked = self.input_layers[i].weight * mask_i.T
            h0_i = F.relu(F.linear(X, weight_masked, self.input_layers[i].bias))
            h1_i = F.relu(self.hidden_layer(h0_i))
            out_i = self.output_layers[i](h1_i)
            outs.append(out_i)
            if i == 0:
                out_0 = out_i

        Out = torch.cat(outs, dim=1)
        return Out, out_0

    def get_W(self) -> torch.Tensor:
        """Compute the weighted adjacency matrix from input-layer weights.

        Returns
        -------
        W : torch.Tensor of shape (num_inputs, num_inputs)
            ``W[j, i]`` is the L2 norm of column *j* in masked weight matrix *i*,
            representing the influence of feature *j* on feature *i*.
        """
        W_cols = []
        for i in range(self.num_inputs):
            mask_i = getattr(self, f"mask_{i}")
            weight_masked = self.input_layers[i].weight * mask_i.T  # (n_hidden, num_inputs)
            norm = torch.norm(weight_masked, p=2, dim=0)  # (num_inputs,)
            W_cols.append(norm.unsqueeze(1))

        return torch.cat(W_cols, dim=1)


class CASTLE(_BaseCausalDiscovery):
    """
    Causal structure learning with CASTLE regularization.

    This class implements the CASTLE algorithm [1]_ for joint causal discovery
    and supervised prediction.  Given a tabular dataset the algorithm learns a
    DAG among all variables while simultaneously training a neural network to
    predict a designated target variable.  The supervised loss is the primary
    objective; DAG-structure recovery acts as a regularizer controlled by
    ``reg_lambda``.

    Parameters
    ----------
    reg_lambda : float, default=1.0
        Weight on the DAG regularization term R_DAG (reconstruction loss,
        acyclicity penalty, and group-lasso sparsity).

    reg_beta : float, default=5.0
        Weight on the group-lasso sparsity term inside R_DAG.

    rho : float, default=1.0
        Initial penalty coefficient for the augmented-Lagrangian acyclicity
        constraint.  Doubled whenever the constraint ``h(W)`` fails to
        decrease by at least a factor of ``0.25`` between epochs.

    lr : float, default=0.001
        Learning rate for the Adam optimizer.

    batch_size : int, default=32
        Mini-batch size for training the neural network.

    n_hidden : int, default=32
        Number of hidden units in each layer of the sub-networks.

    w_threshold : float, default=0.3
        Threshold below which learned adjacency weights are set to zero.

    max_epochs : int, default=200
        Maximum number of training epochs.

    random_state : int | None, default=None
        Seed for reproducible results.  Controls PyTorch weight
        initialization and mini-batch shuffling.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal directed acyclic graph.

    adjacency_matrix_ : pd.DataFrame
        Thresholded weighted adjacency matrix with feature names as both
        the row and column index.

    model_ : _CASTLEModel
        The trained internal PyTorch model instance.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : list[str]
        Names of the independent (non-target) features seen during ``fit``.

    Examples
    --------
    Simulate some data from the *Asia* network and run CASTLE:

    >>> import numpy as np
    >>> from pgmpy.example_models import load_model
    >>> np.random.seed(42)
    >>> model = load_model("bnlearn/asia")
    >>> df = model.simulate(n_samples=500, seed=42)

    >>> from pgmpy.causal_discovery import CASTLE          # doctest: +SKIP
    >>> castle = CASTLE(max_epochs=20, random_state=42)    # doctest: +SKIP
    >>> castle.fit(df)                                     # doctest: +SKIP
    CASTLE(max_epochs=20, random_state=42)
    >>> castle.causal_graph_                               # doctest: +SKIP
    <pgmpy.base.DAG.DAG object at 0x...>

    References
    ----------
    .. [1] Kyono, T., Zhang, Y., & van der Schaar, M. (2020). CASTLE:
           Regularization via Auxiliary Causal Graph Discovery.  *Advances in
           Neural Information Processing Systems 33* (NeurIPS 2020).
    """

    def __init__(
        self,
        reg_lambda: float = 1.0,
        reg_beta: float = 5.0,
        rho: float = 1.0,
        lr: float = 0.001,
        batch_size: int = 32,
        n_hidden: int = 32,
        w_threshold: float = 0.3,
        max_epochs: int = 200,
        random_state: int | None = None,
    ):
        _check_soft_dependencies("torch", obj=self)

        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.rho = rho
        self.lr = lr
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.w_threshold = w_threshold
        self.max_epochs = max_epochs
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        """Fit data to a causal graph.

        Validates ``X`` via the base class, then delegates to :meth:`_fit`.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.  All columns participate in DAG learning.
        y : ignored
            Present for API compatibility with ``sklearn.base.BaseEstimator``.
        **kwargs
            Forwarded to :meth:`_fit`.  Supports ``target_col``.

        Returns
        -------
        self : CASTLE
        """
        X = self._check_fit_data(X)
        return self._fit(X, **kwargs)

    def _fit(self, X: pd.DataFrame, target_col=None):
        """
        Internal fitting procedure for the CASTLE algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            Validated input data (all columns used for DAG learning).
        target_col : str or int, optional
            Column to use as the supervised prediction target.  Defaults to
            the first column.

        Returns
        -------
        self : CASTLE
        """
        cols = list(X.columns)

        if target_col is None:
            target_col = cols[0]
        elif target_col not in cols:
            if isinstance(target_col, int) and target_col < len(cols):
                target_col = cols[target_col]
            else:
                raise ValueError(f"target_col {target_col} not found in columns")

        # Move target to position 0
        cols.remove(target_col)
        cols.insert(0, target_col)
        X_np = X[cols].to_numpy(dtype=np.float32)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_np)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_train = torch.tensor(X_scaled)
        y_train = X_train[:, 0:1]

        num_inputs = X_scaled.shape[1]

        model = _CASTLEModel(num_inputs=num_inputs, n_hidden=self.n_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        rho = float(self.rho)
        alpha = 0.0
        prev_h = float("inf")

        for epoch in range(self.max_epochs):
            model.train()
            n_train = X_train.shape[0]
            indices = torch.randperm(n_train)

            for start_idx in range(0, n_train, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_train)
                batch_idx = indices[start_idx:end_idx]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                optimizer.zero_grad()

                Out, out_0 = model(X_batch)

                supervised_loss = F.mse_loss(out_0, y_batch)

                recon_loss = F.mse_loss(Out, X_batch)

                group_lasso = 0.0
                for i in range(num_inputs):
                    mask_i = getattr(model, f"mask_{i}")
                    w_i = model.input_layers[i].weight * mask_i.T
                    group_lasso += w_i.norm(p=2, dim=0).sum()

                W = model.get_W()
                h = _dag_constraint(W)
                dag_penalty = 0.5 * rho * h * h + alpha * h

                loss = supervised_loss + self.reg_lambda * (recon_loss + dag_penalty + self.reg_beta * group_lasso)

                loss.backward()
                optimizer.step()

                # Re-zero masked positions so node i never sees itself
                with torch.no_grad():
                    for i in range(num_inputs):
                        mask_i = getattr(model, f"mask_{i}")
                        model.input_layers[i].weight.data *= mask_i.T

            model.eval()
            with torch.no_grad():
                W_epoch = model.get_W()
                h_val = _dag_constraint(W_epoch).item()

                # Update augmented-Lagrangian multipliers
                alpha += rho * h_val
                if h_val > 0.25 * prev_h:
                    rho *= 2.0
                prev_h = h_val

        model.eval()
        with torch.no_grad():
            W_final = model.get_W().cpu().numpy()

        W_final[np.abs(W_final) < self.w_threshold] = 0.0

        self.adjacency_matrix_ = pd.DataFrame(W_final, columns=cols, index=cols)

        dag = DAG()
        dag.add_nodes_from(cols)
        for i in range(num_inputs):
            for j in range(num_inputs):
                if W_final[j, i] != 0.0:
                    dag.add_edge(cols[j], cols[i])

        self.causal_graph_ = dag
        self.model_ = model
        self.cols_ = cols
        self.feature_names_in_ = cols[1:]

        return self

    def predict(self, X):
        """Predict the target column using the learned sub-network.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data containing **only** the independent features (i.e.
            without the target column).

        Returns
        -------
        predictions : np.ndarray
            1-D array of predicted values for the target column.
        """
        if isinstance(X, pd.DataFrame):
            X_feats = X[self.feature_names_in_].to_numpy(dtype=np.float32)
        else:
            X_feats = np.asarray(X, dtype=np.float32)

        # Pad the target column (index 0) with zeros so the tensor
        # matches the network's expected input width.
        X_padded = np.zeros((X_feats.shape[0], len(self.cols_)), dtype=np.float32)
        X_padded[:, 1:] = X_feats

        X_scaled = self.scaler_.transform(X_padded)
        X_tensor = torch.tensor(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            _, out_0 = self.model_(X_tensor)

        out_0_np = out_0.cpu().numpy()

        # Inverse-transform only the target column
        dummy = np.zeros((out_0_np.shape[0], len(self.cols_)))
        dummy[:, 0] = out_0_np[:, 0]
        unscaled = self.scaler_.inverse_transform(dummy)

        return unscaled[:, 0]

    def get_adjacency_matrix(self):
        """Return the thresholded adjacency matrix as a ``pd.DataFrame``."""
        return self.adjacency_matrix_

    def get_dag(self):
        """Return the learned DAG (convenience alias for ``causal_graph_``)."""
        return self.causal_graph_
