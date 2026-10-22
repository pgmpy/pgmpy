import copy
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy.parameterization._base import BaseParameterization, _checked_state_names, _states
from pgmpy.parameterization.adapters.pyro_estimators import BasePyroEstimator, PyroSVI
from pgmpy.parameterization.distributions import NominalDistribution, PosteriorPredictive
from pgmpy.parameterization.distributions.posterior_predictive import _unvalidated

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class PyroAdapter(BaseParameterization):
    """Parameterization from a Pyro function: the distribution of the target given its parents, with priors on its
    parameters, fitted by ``estimator``: SVI by default, or MCMC with ``PyroNUTS``.

    ``fn`` takes the parents' values, a dict that maps each parent's name to a 1-D tensor, empty for a root, and returns
    the target's Pyro distribution, with ``event_shape`` () and one batch element per row, or a single one. Numeric
    parents come in torch's default dtype. Discrete parents, those listed in ``state_names`` or that aren't numeric,
    come as integer codes over their states, 0 for the first: the states in ``state_names``, else the sorted states in
    the data. ``fn`` declares its parameters with ``pyro.sample`` and a prior, or with ``pyro.param`` for a point
    estimate, which ``PyroSVI`` fits but ``PyroNUTS`` can't; a vector-valued parameter takes ``.to_event(1)``, not a
    plate. Its sites can have any name but the target's, which ``fit`` gives the target's observed values: in a plate
    over the rows, with each row's log-likelihood multiplied by its weight. The estimator fits this model, and the
    adapter keeps its posterior draws, which every prediction reuses.

    A continuous target, the default, has numeric values, including counts. A discrete target, with
    ``variable_type="discrete"``, has labels, coded like a discrete parent's states, and ``fn`` returns a distribution
    over the codes, e.g. a ``Bernoulli`` for two states or a ``Categorical``. Its ``predict_proba`` gives each row the
    probability of each state averaged over the posterior draws, as a ``NominalDistribution``.

    When predicting, every latent value gets the posterior draws as a new first dimension. A scalar becomes
    ``(draws, 1)``, so it broadcasts with the parents. A parameter with event dimensions, e.g. a weight vector of shape
    ``(k,)``, becomes ``(draws, k)``: combine it with the parents through ``torch.einsum`` with ``...``, e.g.
    ``torch.einsum("nk,...k->...n", features, weights)``.

    Each node keeps its Pyro parameters in a store of its own, so nodes with the same parameter names don't overwrite
    each other, and the global store is left as it was. Neither fitting nor prediction changes the global torch random
    state.

    Inference converges much faster when ``fn`` centers and scales the parents, and puts its priors on the data's scale:
    on poorly scaled parents, SVI can stop far from the posterior, and NUTS takes far longer. ``diagnostics_`` shows how
    the inference went, e.g. whether SVI's loss has settled.

    Requires the optional dependencies ``pyro-ppl`` and ``skpro``.

    Parameters
    ----------
    fn : callable
        The target's distribution given its parents: takes the parents' values, a dict of 1-D tensors by name, and
        returns a Pyro distribution.
    variable_type : {"continuous", "discrete"}, default="continuous"
        Whether the target has numeric values or labels.
    state_names : dict, optional
        States of a discrete target and of discrete parents, as ``{variable: [states]}``, in the order of their codes.
        The listed states must include every state in the data; discrete variables that aren't listed get the sorted
        states seen in the data.
    estimator : BasePyroEstimator, optional
        The inference algorithm and its settings, e.g. ``PyroNUTS(num_samples=500)``. ``None`` for ``PyroSVI()``.

    Attributes
    ----------
    posterior_samples_ : dict of str to torch.Tensor
        The estimator's draws of each of ``fn``'s latent sites, along the first dimension; empty without latent sites.
    params_ : dict of str to torch.Tensor
        The fitted values of ``fn``'s ``pyro.param`` sites.
    diagnostics_ : dict
        What the estimator reports about the inference, e.g. ``{"losses": ...}`` for ``PyroSVI``.
    state_names_ : dict
        States of a discrete target and of the discrete parents, in the order of their codes.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pyro
    >>> import pyro.distributions as dist
    >>> from pgmpy.parameterization import PyroAdapter, PyroSVI
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame({"temp": rng.normal(20, 5, size=300)})
    >>> y = pd.Series(10 + 2 * X["temp"] + rng.normal(scale=3, size=300), name="sales")

    ``fn`` centers and scales the parent, and puts its priors on the data's scale:

    >>> def sales(parents):
    ...     intercept = pyro.sample("intercept", dist.Normal(50.0, 20.0))
    ...     coef = pyro.sample("coef", dist.Normal(0.0, 20.0))
    ...     sigma = pyro.sample("sigma", dist.HalfNormal(10.0))
    ...     return dist.Normal(intercept + coef * (parents["temp"] - 20.0) / 5.0, sigma)
    >>> cpd = PyroAdapter(sales, estimator=PyroSVI(random_state=0)).fit(X, y)
    >>> round(cpd.posterior_samples_["coef"].mean().item())
    10
    >>> new = pd.DataFrame({"temp": [15.0, 25.0]})
    >>> cpd.predict(new).round()["sales"].tolist()
    [40.0, 60.0]
    >>> type(cpd.predict_proba(new)).__name__
    'PosteriorPredictive'

    A discrete target has labels, and ``fn`` returns a distribution over their codes: here a logistic regression, with
    "dry" coded 0 and "wet" 1.

    >>> ground = pd.Series(np.where(y > 50, "wet", "dry"), name="ground")
    >>> def wet(parents):
    ...     intercept = pyro.sample("intercept", dist.Normal(0.0, 2.0))
    ...     coef = pyro.sample("coef", dist.Normal(0.0, 2.0))
    ...     return dist.Bernoulli(logits=intercept + coef * (parents["temp"] - 20.0) / 5.0)
    >>> cpd = PyroAdapter(wet, estimator=PyroSVI(random_state=0), variable_type="discrete").fit(X, ground)
    >>> list(cpd.predict_proba(new).categories)
    ['dry', 'wet']
    >>> cpd.predict(new)["ground"].tolist()
    ['dry', 'wet']
    """

    _tags = {
        "name": "pyro_adapter",
        "variable_type": ["discrete", "continuous"],
        "parent_data_types": ["discrete", "continuous", "mixed"],
        "supports_weighted_data": True,
        "python_dependencies": ["pyro-ppl", "skpro"],
    }

    def __init__(
        self,
        fn: Callable[[dict], Any],
        variable_type: Literal["continuous", "discrete"] = "continuous",
        state_names: dict | None = None,
        estimator: BasePyroEstimator | None = None,
    ) -> None:
        self.fn = fn
        self.variable_type = variable_type
        self.state_names = state_names
        self.estimator = estimator
        super().__init__()

        if not callable(fn):
            raise TypeError(f"fn must be callable, but is a {type(fn).__name__}.")
        if variable_type not in ("continuous", "discrete"):
            raise ValueError(f"variable_type must be 'continuous' or 'discrete', but is {variable_type!r}.")
        self.set_tags(variable_type=[variable_type])
        self._check_estimator(estimator, BasePyroEstimator)

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Find the states of a discrete target and of the discrete parents, those listed in state_names or that
        # aren't numeric: the given states, else the sorted states in the data, also of rows with weight 0.
        given = _checked_state_names(self.state_names)
        discrete = self.get_tag("variable_type") == ["discrete"]
        if not discrete and self.variable_ in given:
            raise ValueError(
                f"state_names lists {self.variable_!r}, a continuous target, but only discrete variables have states."
            )
        columns = [(self.variable_, y.iloc[:, 0])] if discrete else []
        columns += [
            (name, X.iloc[:, position])
            for position, name in enumerate(self.evidence_)
            if name in given or not pd.api.types.is_numeric_dtype(X.iloc[:, position])
        ]
        self.state_names_ = _states(columns, given)

        # Step 2: Drop the rows with weight 0, which count for nothing, as Pyro only scales by positive weights.
        if sample_weight is not None:
            keep = sample_weight > 0
            X, y, sample_weight = X[keep], y[keep], sample_weight[keep]

        # Step 3: Convert the data to tensors, a discrete target to the codes of its states, and build the node's model
        # around fn.
        dtype = torch.get_default_dtype()
        parents = self._tensors(X)
        if discrete:
            codes = pd.Index(self.state_names_[self.variable_], tupleize_cols=False).get_indexer(y.iloc[:, 0])
            if (codes < 0).any():
                raise ValueError(
                    "Some values of y don't match state_names, e.g. [0, 1] given for boolean data; list the states "
                    "with the data's own types."
                )
            target = torch.as_tensor(codes, dtype=dtype)
        else:
            target = torch.as_tensor(y.iloc[:, 0].to_numpy(dtype=float), dtype=dtype)
        weights = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=dtype)
        model = self._model(len(y))

        # Step 4: Work in a parameter store of the node's own, so that neither the user's global store nor other nodes
        # change. Trace fn once, in a copy of the torch random state, to check the distribution it returns and find its
        # latent and param sites.
        with pyro.get_param_store().scope() as state:
            with torch.random.fork_rng(devices=[]):
                trace = pyro.poutine.trace(self.fn).get_trace(parents)
            distribution = trace.nodes["_RETURN"]["value"]
            site = str(self.variable_)
            if not isinstance(distribution, torch.distributions.Distribution):
                raise TypeError(
                    f"fn must return a Pyro distribution of the target, but returned a {type(distribution).__name__}."
                )
            if distribution.event_shape != ():
                raise ValueError(
                    "fn must return a distribution of one value per row, with event_shape (), but its event_shape is "
                    f"{tuple(distribution.event_shape)}."
                )
            if tuple(distribution.batch_shape) not in ((), (1,), (len(y),)):
                raise ValueError(
                    f"fn's distribution must have one batch element per row, {len(y)}, or a single one, but has batch "
                    f"shape {tuple(distribution.batch_shape)}."
                )
            if site in trace.nodes:
                raise ValueError(
                    f"fn has a site named {site!r}, the target's name, which PyroAdapter gives the target's values: "
                    "rename it."
                )
            # A discrete target's distribution must put all its probability on the codes of its states.
            if discrete:
                n_states = len(self.state_names_[self.variable_])
                if not distribution.support.is_discrete:
                    raise ValueError(
                        f"fn must return a distribution with a discrete support over the codes 0 to {n_states - 1} of "
                        "the target's states, e.g. a Categorical or a Bernoulli."
                    )
                total = _code_probabilities(distribution, n_states).sum(-1)
                if not torch.allclose(total, torch.ones_like(total), atol=1e-4):
                    raise ValueError(
                        f"fn's distribution must put all its probability on the codes 0 to {n_states - 1} of the "
                        f"target's states, {self.state_names_[self.variable_]}, but puts "
                        f"{total.min().item():.3g} there."
                    )
            outside = (~distribution.support.check(target)).cpu().numpy()
            if outside.any():
                raise ValueError(
                    "y has values outside the support of fn's distribution, e.g. "
                    f"{y.iloc[:, 0].to_numpy()[outside][0]}."
                )
            latent = [
                name for name, node in trace.nodes.items() if node["type"] == "sample" and not node["is_observed"]
            ]
            params = [name for name, node in trace.nodes.items() if node["type"] == "param"]

            # Step 5: Fit with the estimator, and keep its posterior draws of the latent sites.
            estimator = PyroSVI() if self.estimator is None else self.estimator
            draws, self.diagnostics_ = estimator.estimate(model, (parents, target, weights), latent, params)
            missing = [name for name in latent if name not in draws]
            if missing:
                raise ValueError(f"{type(estimator).__name__} returned no draws of {missing}, latent sites of fn.")
            self.posterior_samples_ = {name: draws[name].detach() for name in latent}
            self.params_ = {name: pyro.param(name).detach() for name in params}
        self._param_state = state
        self._discrete_support = distribution.support.is_discrete

    def _predict_proba(self, X: pd.DataFrame | None) -> PosteriorPredictive | NominalDistribution:
        # Step 1: A continuous target gets the posterior predictive, of a copy of this adapter, so that it keeps this
        # fit after a refit or a set_params. Only a root gets X=None: its own distribution is a scalar one, evaluated on
        # one row without columns.
        if self.get_tag("variable_type") == ["continuous"]:
            fitted = copy.copy(self)
            if X is None:
                return PosteriorPredictive(fitted, pd.DataFrame(index=[0]))
            return PosteriorPredictive(fitted, X, index=X.index, columns=[self.variable_])

        # Step 2: A discrete target gets each row's probabilities of its states, averaged over the posterior draws. A
        # plate of size 0 raises in Pyro, so no rows give no probabilities.
        states = list(self.state_names_[self.variable_])
        rows = pd.DataFrame(index=[0]) if X is None else X
        if len(rows) == 0:
            probs = np.empty((0, len(states)))
        else:
            distribution = self._distribution(rows, self.posterior_samples_)
            probs = _code_probabilities(distribution, len(states)).mean(dim=0).detach().cpu().numpy().astype(float)
        if X is None:
            return NominalDistribution(probs=probs[0], categories=states)
        return NominalDistribution(probs=probs, categories=states, index=X.index, columns=[self.variable_])

    def _distribution(self, X: pd.DataFrame, draws: dict) -> Any:
        """Return the target's torch distribution given the parents' values ``X`` under each of ``draws``, a dict of
        tensors with the draws along the first dimension: batch shape (draws, rows), with one draw without latent sites.
        The node's own store gives fn its pyro.param values, and the global random state doesn't change."""
        n_draws = len(next(iter(draws.values()))) if draws else 1
        with pyro.get_param_store().scope(self._param_state), torch.random.fork_rng(devices=[]):
            predictive = pyro.infer.Predictive(self._model(len(X)), posterior_samples=draws, num_samples=n_draws)
            trace = predictive.get_vectorized_trace(self._tensors(X))
        return trace.nodes[str(self.variable_)]["fn"]

    def _tensors(self, X: pd.DataFrame) -> dict:
        """Return the parents' values as a dict of 1-D tensors by name: integer codes over their states for the discrete
        parents, and floats in torch's default dtype for the others."""
        tensors = {}
        for position, name in enumerate(self.evidence_):
            values = X.iloc[:, position]
            if name in self.state_names_:
                codes = pd.Index(self.state_names_[name], tupleize_cols=False).get_indexer(values)
                if (codes < 0).any():
                    raise ValueError(
                        f"X[{name!r}] has values not seen in fit, or not among its states in state_names: "
                        f"{pd.unique(values.to_numpy()[codes < 0]).tolist()}. Known states: {self.state_names_[name]}."
                    )
                tensors[name] = torch.as_tensor(codes, dtype=torch.long)
            else:
                tensors[name] = torch.as_tensor(values.to_numpy(dtype=float), dtype=torch.get_default_dtype())
        return tensors

    def _model(self, n_rows: int) -> Callable:
        """Return the node's Pyro model: fn's distribution, observed at a site named after the target in a plate over
        the rows, with each row's log-likelihood multiplied by its weight."""
        site = str(self.variable_)

        def model(parents: dict, target: Any = None, weights: Any = None) -> Any:
            distribution = self.fn(parents)
            with pyro.plate(f"{site}_rows", n_rows), pyro.poutine.scale(scale=1.0 if weights is None else weights):
                return pyro.sample(site, distribution, obs=target)

        return model


def _code_probabilities(distribution: Any, n_states: int) -> Any:
    """Return the probability of each code 0, ..., n_states - 1 under the distribution, along a new last dimension: 0
    for codes outside its support, where torch would reject them."""
    codes = torch.arange(n_states, dtype=torch.get_default_dtype()).reshape(
        n_states, *[1] * len(distribution.batch_shape)
    )
    inside = distribution.support.check(codes)
    return torch.where(inside, _unvalidated(distribution).log_prob(codes).exp(), 0.0).movedim(0, -1)
