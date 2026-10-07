import copy
from collections.abc import Callable
from numbers import Integral
from typing import Any, Literal

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy import config
from pgmpy.parameterization._base import BaseParameterization
from pgmpy.parameterization.distributions import PosteriorPredictive

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class PyroAdapter(BaseParameterization):
    """Parameterization from a Pyro function: the distribution of the target given its parents, with priors on its
    parameters, fitted by SVI or by MCMC.

    ``fn`` takes the parents' values, a dict that maps each parent's name to a 1-D tensor in torch's default dtype,
    empty for a root, and returns the target's Pyro distribution, with ``event_shape`` () and one batch element per row,
    or a single one. It declares its parameters with ``pyro.sample`` and a prior, or with ``pyro.param`` for a point
    estimate, which only SVI fits; a vector-valued parameter takes ``.to_event(1)``, not a plate. Its sites can have any
    name but the target's, which ``fit`` gives the target's observed values: in a plate over the rows, with each row's
    log-likelihood multiplied by its weight. ``fit`` fits this model by SVI, with an ``AutoMultivariateNormal`` guide by
    default, or by NUTS, and keeps ``num_samples`` posterior draws, which every prediction reuses.

    Each node keeps its Pyro parameters in a store of its own, so nodes with the same parameter names don't overwrite
    each other, and the global store is left as it was. Fitting draws from a copy of the torch random state, seeded from
    ``random_state``, so the global random state doesn't change either.

    Inference converges much faster when ``fn`` centers and scales the parents, and puts its priors on the data's scale:
    on poorly scaled parents, SVI can stop far from the posterior, and NUTS takes far longer. ``losses_`` shows whether
    SVI's loss has settled.

    Requires the optional dependencies ``pyro-ppl`` and ``skpro``.

    Parameters
    ----------
    fn : callable
        The target's distribution given its parents: takes the parents' values, a dict of 1-D tensors by name, and
        returns a Pyro distribution.
    inference : {"svi", "mcmc"}, default="svi"
        Stochastic variational inference, or MCMC with NUTS.
    num_samples : int, default=1000
        Number of posterior draws to keep: drawn from the guide after SVI, or NUTS samples after as many warmup steps.
    num_steps : int, default=1000
        Number of SVI steps. Not used by MCMC.
    guide : callable, optional
        Takes the model and returns a guide, e.g. ``pyro.infer.autoguide.AutoNormal``. ``None`` for
        ``AutoMultivariateNormal``. Not used by MCMC, nor when ``fn`` has no latent sites.
    optim : pyro.optim.PyroOptim, optional
        Optimizer for SVI, copied for each fit, so the object passed in doesn't change. ``None`` for
        ``pyro.optim.ClippedAdam``, with a learning rate that decays from 0.1 to 0.001 over the steps. Not used by MCMC.
    random_state : int, numpy.random.Generator or None, optional
        Seed or generator for fitting; ``None`` draws fresh values.

    Attributes
    ----------
    posterior_samples_ : dict of str to torch.Tensor
        ``num_samples`` draws of each of ``fn``'s latent sites, along the first dimension; empty without latent sites.
    params_ : dict of str to torch.Tensor
        The fitted values of ``fn``'s ``pyro.param`` sites.
    losses_ : numpy.ndarray of shape (num_steps,)
        SVI's loss after each step. Only for SVI.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pyro
    >>> import pyro.distributions as dist
    >>> from pgmpy.parameterization import PyroAdapter
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame({"temp": rng.normal(20, 5, size=300)})
    >>> y = pd.Series(10 + 2 * X["temp"] + rng.normal(scale=3, size=300), name="sales")

    ``fn`` centers and scales the parent, and puts its priors on the data's scale:

    >>> def sales(parents):
    ...     intercept = pyro.sample("intercept", dist.Normal(50.0, 20.0))
    ...     coef = pyro.sample("coef", dist.Normal(0.0, 20.0))
    ...     sigma = pyro.sample("sigma", dist.HalfNormal(10.0))
    ...     return dist.Normal(intercept + coef * (parents["temp"] - 20.0) / 5.0, sigma)
    >>> cpd = PyroAdapter(sales, random_state=0).fit(X, y)
    >>> round(cpd.posterior_samples_["coef"].mean().item())
    10
    >>> new = pd.DataFrame({"temp": [15.0, 25.0]})
    >>> cpd.predict(new).round()["sales"].tolist()
    [40.0, 60.0]
    >>> type(cpd.predict_proba(new)).__name__
    'PosteriorPredictive'
    """

    _tags = {
        "name": "pyro_adapter",
        "variable_type": ["continuous"],
        "parent_data_types": ["continuous"],
        "supports_weighted_data": True,
        "python_dependencies": ["pyro-ppl", "skpro"],
    }

    def __init__(
        self,
        fn: Callable[[dict], Any],
        inference: Literal["svi", "mcmc"] = "svi",
        num_samples: int = 1000,
        num_steps: int = 1000,
        guide: Callable | None = None,
        optim: Any | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.fn = fn
        self.inference = inference
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.guide = guide
        self.optim = optim
        self.random_state = random_state
        super().__init__()

        if not callable(fn):
            raise TypeError(f"fn must be callable, but is a {type(fn).__name__}.")
        if inference not in ("svi", "mcmc"):
            raise ValueError(f"inference must be 'svi' or 'mcmc', but is {inference!r}.")
        for name, value in (("num_samples", num_samples), ("num_steps", num_steps)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but is {value!r}.")

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        # Step 1: Drop the rows with weight 0, which count for nothing, as Pyro only scales by positive weights. Convert
        # the data to tensors, and build the node's model around fn.
        if sample_weight is not None:
            keep = sample_weight > 0
            X, y, sample_weight = X[keep], y[keep], sample_weight[keep]
        dtype = torch.get_default_dtype()
        parents = self._tensors(X)
        target = torch.as_tensor(y.iloc[:, 0].to_numpy(dtype=float), dtype=dtype)
        weights = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=dtype)
        model = self._model(len(y))

        # Step 2: Work in a parameter store of the node's own, and in a copy of the torch random state seeded from
        # random_state, so that neither the user's global store nor the global random state changes.
        seed = int(np.random.default_rng(self.random_state).integers(2**63))
        with pyro.get_param_store().scope() as state, torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            # Step 3: Trace fn once, to check the distribution it returns and find its latent and param sites.
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
            outside = ~distribution.support.check(target)
            if outside.any():
                raise ValueError(
                    f"y has values outside the support of fn's distribution, e.g. {target[outside][0].item()}."
                )
            latent = [
                name for name, node in trace.nodes.items() if node["type"] == "sample" and not node["is_observed"]
            ]
            params = [name for name, node in trace.nodes.items() if node["type"] == "param"]

            # Step 4: Fit, and keep num_samples posterior draws of the latent sites. Without latent sites the guide is
            # empty, as Pyro's continuous autoguides need one, and SVI fits fn's pyro.param values by maximum
            # likelihood.
            if self.inference == "svi":
                if latent:
                    guide = (pyro.infer.autoguide.AutoMultivariateNormal if self.guide is None else self.guide)(model)
                else:
                    guide = _empty_guide
                # By default the learning rate decays from 0.1 to 0.001 over the steps: large steps reach the posterior
                # from the priors, and small ones settle in it.
                if self.optim is None:
                    optim = pyro.optim.ClippedAdam({"lr": 0.1, "lrd": 0.01 ** (1 / self.num_steps)})
                else:
                    optim = copy.deepcopy(self.optim)
                svi = pyro.infer.SVI(model, guide, optim, pyro.infer.Trace_ELBO())
                self.losses_ = np.array([svi.step(parents, target, weights) for _ in range(self.num_steps)])
                if not np.isfinite(self.losses_[-1]):
                    raise ValueError(
                        "SVI's loss isn't finite after the last step: lower the learning rate of optim, or check that "
                        "fn's priors are on the data's scale."
                    )
                draws = pyro.infer.Predictive(guide, num_samples=self.num_samples)(parents, target, weights)
            else:
                if params:
                    raise ValueError(
                        f"NUTS can't fit pyro.param values, but fn has {params}: give them priors with pyro.sample, or "
                        "use inference='svi'."
                    )
                if not latent:
                    raise ValueError(
                        "NUTS needs a latent site, a parameter with a prior from pyro.sample, but fn has none."
                    )
                mcmc = pyro.infer.MCMC(
                    pyro.infer.NUTS(model), num_samples=self.num_samples, disable_progbar=not config.SHOW_PROGRESS
                )
                mcmc.run(parents, target, weights)
                draws = mcmc.get_samples()
            self.posterior_samples_ = {name: draws[name].detach() for name in latent}
            self.params_ = {name: pyro.param(name).detach() for name in params}
        self._param_state = state
        self._discrete = distribution.support.is_discrete

    def _predict_proba(self, X: pd.DataFrame | None) -> PosteriorPredictive:
        # Only a root gets X=None: its own distribution is a scalar one, evaluated on one row without columns.
        if X is None:
            return PosteriorPredictive(self, pd.DataFrame(index=[0]))
        return PosteriorPredictive(self, X, index=X.index, columns=[self.variable_])

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
        """Return the parents' values as a dict of 1-D tensors by name, in torch's default dtype."""
        dtype = torch.get_default_dtype()
        return {
            name: torch.as_tensor(X.iloc[:, position].to_numpy(dtype=float), dtype=dtype)
            for position, name in enumerate(self.evidence_)
        }

    def _model(self, n_rows: int) -> Callable:
        """Return the node's Pyro model: fn's distribution, observed at a site named after the target in a plate over
        the rows, with each row's log-likelihood multiplied by its weight."""
        site = str(self.variable_)

        def model(parents: dict, target: Any = None, weights: Any = None) -> Any:
            distribution = self.fn(parents)
            with pyro.plate(f"{site}_rows", n_rows), pyro.poutine.scale(scale=1.0 if weights is None else weights):
                return pyro.sample(site, distribution, obs=target)

        return model


def _empty_guide(*args: Any) -> None:
    """The guide of a model without latent sites, which SVI fits by maximum likelihood."""
