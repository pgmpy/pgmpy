import math
from collections.abc import Callable
from contextlib import nullcontext

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.base import BaseEstimator, clone

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.bivariate_scores import BaseBivariateScore, get_bivariate_score
from pgmpy.utils import get_dataset_type

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


class NormFlow(nn.Module):
    """
    Disturbance estimator for the Post-Nonlinear model.

    For data in ``[cause, effect]`` order, the estimator learns an inner
    function of the cause and a monotone transformation of the effect. Their
    difference is the estimated disturbance. This estimator requires PyTorch.

    Parameters
    ----------
    hidden_dim : int, default=12
        Number of hidden units in both learned functions.

    n_components : int, default=5
        Number of Gaussian-mixture components used for the disturbance.

    max_iter : int, default=1000
        Number of full-batch optimization steps.

    learning_rate : float, default=1e-3
        Adam learning rate.

    seed : int, optional
        PyTorch random seed.
    """

    def __init__(self, hidden_dim=12, n_components=5, max_iter=1000, learning_rate=1e-3, seed=None):
        super().__init__()
        _check_soft_dependencies("torch", obj=self)

        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.seed = seed

        if self.hidden_dim < 1 or self.n_components < 1 or self.max_iter < 1:
            raise ValueError("hidden_dim, n_components, and max_iter must all be positive integers.")

        seed_context = torch.random.fork_rng(devices=[]) if self.seed is not None else nullcontext()
        with seed_context:
            if self.seed is not None:
                torch.manual_seed(self.seed)

            self.inner_model_ = nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.post_input_weight_ = nn.Parameter(torch.zeros(1, self.hidden_dim))
            self.post_output_weight_ = nn.Parameter(torch.zeros(1, self.hidden_dim))
            # Distinct biases prevent hidden units from receiving identical gradients.
            self.post_bias_ = nn.Parameter(torch.linspace(-1.0, 1.0, self.hidden_dim).reshape(1, -1))
            self.post_slope_ = nn.Parameter(torch.zeros(1, 1))
            self.post_intercept_ = nn.Parameter(torch.zeros(1, 1))
            self.noise_logits_ = nn.Parameter(torch.zeros(self.n_components))
            self.noise_mean_ = nn.Parameter(torch.linspace(-1.0, 1.0, self.n_components))
            self.noise_scale_ = nn.Parameter(torch.zeros(self.n_components))

        self.register_buffer("mean_", torch.zeros(2))
        self.register_buffer("scale_", torch.ones(2))
        self.register_buffer("fitted_", torch.tensor(False))

    def _post_transform(self, effect):
        """Evaluate the monotone post-transform and its positive derivative."""
        input_weight = torch.nn.functional.softplus(self.post_input_weight_)
        output_weight = torch.nn.functional.softplus(self.post_output_weight_)
        slope = torch.nn.functional.softplus(self.post_slope_) + 1e-4

        hidden = torch.tanh(effect * input_weight + self.post_bias_)
        transformed = self.post_intercept_ + slope * effect + hidden @ output_weight.T
        derivative = slope + ((1.0 - hidden**2) * input_weight * output_weight).sum(dim=1, keepdim=True)
        return transformed, derivative

    def _negative_log_density(self, disturbance):
        """Evaluate the learned Gaussian-mixture negative log density."""
        scale = torch.nn.functional.softplus(self.noise_scale_) + 1e-4
        standardized = (disturbance - self.noise_mean_) / scale

        component_log_prob = -0.5 * standardized**2 - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
        log_weight = torch.log_softmax(self.noise_logits_, dim=0)
        return -torch.logsumexp(component_log_prob + log_weight, dim=1).mean()

    def forward(self, X):
        """Estimate disturbances for a two-column tensor."""
        if not self.fitted_.item():
            raise RuntimeError("NormFlow must be fitted before calling forward.")

        X = torch.as_tensor(X, dtype=self.mean_.dtype, device=self.mean_.device)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"NormFlow expects a 2-column tensor, got shape {tuple(X.shape)}.")

        standardized = (X - self.mean_) / self.scale_
        cause = standardized[:, 0:1]
        effect = standardized[:, 1:2]
        transformed_effect, _ = self._post_transform(effect)
        return transformed_effect - self.inner_model_(cause)

    def fit(self, X, y=None):
        """
        Fit the disturbance estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Data in ``[cause, effect]`` column order.

        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : NormFlow
            The fitted estimator.
        """
        X = np.array(X, dtype=float, copy=True, order="C")
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"NormFlow expects a 2-column array, got shape {X.shape}.")

        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        if np.any(scale == 0):
            raise ValueError("NormFlow requires non-constant cause and effect columns.")

        with torch.no_grad():
            self.mean_.copy_(torch.as_tensor(mean, dtype=self.mean_.dtype, device=self.mean_.device))
            self.scale_.copy_(torch.as_tensor(scale, dtype=self.scale_.dtype, device=self.scale_.device))
            self.fitted_.fill_(False)

        X_t = torch.tensor(X, dtype=self.mean_.dtype, device=self.mean_.device)
        standardized = (X_t - self.mean_) / self.scale_
        cause_t = standardized[:, 0:1]
        effect_t = standardized[:, 1:2]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            transformed_effect, derivative = self._post_transform(effect_t)
            disturbance = transformed_effect - self.inner_model_(cause_t)
            loss = self._negative_log_density(disturbance) - torch.log(derivative).mean()
            if not torch.isfinite(loss):
                raise ValueError("PNL optimization produced a non-finite loss.")
            loss.backward()
            optimizer.step()

        self.loss_ = float(loss.detach())
        self.fitted_.fill_(True)
        return self

    def predict(self, X):
        """
        Estimate the PNL disturbance for each observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Candidate cause and effect observations.

        Returns
        -------
        disturbance : np.ndarray of shape (n_samples,)
            Estimated disturbance values.
        """
        X = np.array(X, dtype=float, copy=True, order="C")
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"NormFlow expects a 2-column array, got shape {X.shape}.")

        X_t = torch.tensor(X, dtype=self.mean_.dtype, device=self.mean_.device)
        with torch.no_grad():
            disturbance = self(X_t)
        return disturbance.cpu().numpy().ravel()


class PNL(BaseCausalDiscovery):
    """
    Bivariate causal discovery using the Post-Nonlinear model
    :cite:p:`zhang_hyvarinen_2009`.

    PNL assumes ``effect = f2(f1(cause) + disturbance)``, where ``f2`` is
    invertible and the disturbance is independent of the cause. The method fits
    both directions and selects the direction with the smaller dependence
    score. Equal or non-finite scores cannot determine a direction.

    Parameters
    ----------
    estimator : sklearn-compatible estimator, default=None
        Estimator used to recover the disturbance. It must be cloneable, accept
        data in ``[cause, effect]`` order in ``fit``, and return one disturbance
        value per sample from ``predict``. If ``None``, :class:`NormFlow` is
        used.

    scoring_method : str, BaseBivariateScore instance, or callable, default="independence"
        Score used to measure dependence between the candidate cause and the
        estimated disturbance. A smaller score indicates the preferred
        direction.

    seed : int, optional
        Random seed passed to the default estimator. Ignored when ``estimator``
        is provided.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        Learned graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of ``causal_graph_``.

    forward_score_ : float
        Score for the first-column to second-column direction.

    backward_score_ : float
        Score for the second-column to first-column direction.

    forward_estimator_ : estimator
        Fitted disturbance estimator for the forward direction.

    backward_estimator_ : estimator
        Fitted disturbance estimator for the backward direction.

    n_features_in_ : int
        Number of variables in the data.

    feature_names_in_ : np.ndarray
        Variable names in the data.

    References
    ----------
    - :cite:p:`zhang_hyvarinen_2009`
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        scoring_method: str
        | BaseBivariateScore
        | Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float] = "independence",
        seed: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.scoring_method = scoring_method
        self.seed = seed

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame) -> "PNL":
        """
        Fit both candidate directions and orient the edge.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing exactly two continuous variables.

        Returns
        -------
        self : PNL
            Fitted estimator.
        """
        # Step 1: Validate the input data.
        if X.shape[1] != 2:
            raise ValueError(f"PNL requires exactly two variables, got {X.shape[1]}.")
        if get_dataset_type(X) != "continuous":
            raise ValueError("PNL requires continuous (numeric) variables; got non-continuous data.")

        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; PNL requires non-constant variables.")

        # Step 2: Fit and score both directions.
        score_fn = get_bivariate_score(self.scoring_method, algorithm="pnl")
        forward_score, forward_estimator = self._direction_score(X[[x, y]], score_fn)
        backward_score, backward_estimator = self._direction_score(X[[y, x]], score_fn)

        # Step 3: Reject ambiguous results before storing fitted attributes.
        if forward_score == backward_score:
            raise ValueError(
                "PNL could not determine a causal direction because both directions produced the same score: "
                f"{forward_score!r}."
            )

        self.forward_score_ = forward_score
        self.backward_score_ = backward_score
        self.forward_estimator_ = forward_estimator
        self.backward_estimator_ = backward_estimator

        # Step 4: Orient the edge toward the direction with the smaller score.
        edge = (x, y) if self.forward_score_ < self.backward_score_ else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")
        return self

    def _direction_score(
        self,
        direction_data: pd.DataFrame,
        score_fn: Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float],
    ) -> tuple[float, BaseEstimator | nn.Module]:
        """Fit and score one candidate direction."""
        if self.estimator is None:
            estimator = NormFlow(seed=self.seed)
        else:
            estimator = clone(self.estimator)

        estimator.fit(direction_data)
        disturbance = np.asarray(estimator.predict(direction_data)).ravel()
        if not np.isfinite(disturbance).all():
            raise ValueError("PNL estimator produced non-finite disturbance predictions.")

        cause = direction_data.iloc[:, 0].to_numpy()
        direction_score = float(score_fn(cause, disturbance))
        if not np.isfinite(direction_score):
            raise ValueError("PNL dependence score is non-finite.")
        return direction_score, estimator
