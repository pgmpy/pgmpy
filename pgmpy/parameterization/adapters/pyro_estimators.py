import copy
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
from skbase.utils.dependencies import _safe_import

from pgmpy import config
from pgmpy.parameterization._base import BaseLocalEstimator

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class BasePyroEstimator(BaseLocalEstimator):
    """Base class for the estimators of a ``PyroAdapter``'s posterior: the inference algorithm.

    ``PyroAdapter.fit`` builds the node's Pyro model, finds the latent and param sites of its function, and passes them
    to ``estimate`` inside a parameter store of the node's own, so that any ``pyro.param`` values the estimator fits
    stay with the node. ``estimate`` returns posterior draws and its own diagnostics, and must leave the global torch
    random state as it was. Subclasses implement ``estimate``, and set a ``name``.
    """

    _tags = {"parameterization": ["pyro_adapter"], "supports_weighted_data": True, "python_dependencies": "pyro-ppl"}

    def estimate(
        self, model: Callable, args: tuple, latent: list[str], params: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fit the node's model, and return posterior draws and diagnostics.

        Parameters
        ----------
        model : callable
            The node's Pyro model. Its sample weights scale each row's log-likelihood, so the estimator needs no
            weights of its own.
        args : tuple
            The model's arguments: the parents' values, the target's values and the weights.
        latent : list of str
            Names of the function's latent sites, those with a prior from ``pyro.sample``.
        params : list of str
            Names of the function's ``pyro.param`` sites.

        Returns
        -------
        draws : dict of str to torch.Tensor
            Posterior draws of at least the latent sites, with the draws along the first dimension.
        diagnostics : dict
            Whatever the estimator reports about the inference, documented by each estimator.
        """
        raise NotImplementedError


class PyroSVI(BasePyroEstimator):
    """Stochastic variational inference of a ``PyroAdapter``'s posterior.

    SVI fits a guide, an ``AutoMultivariateNormal`` by default, together with the function's ``pyro.param`` values, and
    then draws ``num_samples`` posterior samples from the guide. Without latent sites the guide is empty, and SVI fits
    the ``pyro.param`` values by maximum likelihood. Its diagnostics are ``{"losses": numpy.ndarray}``, the loss after
    each step, which shows whether SVI has settled.

    Parameters
    ----------
    num_steps : int, default=1000
        Number of SVI steps.
    num_samples : int, default=1000
        Number of posterior draws to keep.
    guide : callable, optional
        Takes the model and returns a guide, e.g. ``pyro.infer.autoguide.AutoNormal``. ``None`` for
        ``AutoMultivariateNormal``. Not used without latent sites.
    optim : pyro.optim.PyroOptim, optional
        Optimizer, copied for each fit, so the object passed in doesn't change. ``None`` for ``pyro.optim.ClippedAdam``,
        with a learning rate that decays from 0.1 to 0.001 over the steps.
    random_state : int, numpy.random.Generator or None, optional
        Seed or generator for fitting; ``None`` draws fresh values. The global torch random state doesn't change.

    Examples
    --------
    With a N(0, 10^2) prior on the mean and four observations summing to 8, the posterior mean is 8 / 4.01:

    >>> import pandas as pd
    >>> import pyro
    >>> import pyro.distributions as dist
    >>> from pgmpy.parameterization import PyroAdapter, PyroSVI
    >>> def unknown_mean(parents):
    ...     return dist.Normal(pyro.sample("mu", dist.Normal(0.0, 10.0)), 1.0)
    >>> y = pd.Series([1.0, 2.0, 3.0, 2.0], name="y")
    >>> node = PyroAdapter(unknown_mean, estimator=PyroSVI(num_steps=500, random_state=0)).fit(None, y)
    >>> round(node.posterior_samples_["mu"].mean().item())
    2
    >>> node.diagnostics_["losses"].shape
    (500,)
    """

    _tags = {"name": "pyro_svi"}

    def __init__(
        self,
        num_steps: int = 1000,
        num_samples: int = 1000,
        guide: Callable | None = None,
        optim: Any | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.guide = guide
        self.optim = optim
        self.random_state = random_state
        super().__init__()

        _check_positive_integer("num_steps", num_steps)
        _check_positive_integer("num_samples", num_samples)

    def estimate(
        self, model: Callable, args: tuple, latent: list[str], params: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seed = int(np.random.default_rng(self.random_state).integers(2**63))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            # Step 1: Fit the guide and the pyro.param values. Without latent sites the guide is empty, as Pyro's
            # continuous autoguides need one. By default the learning rate decays from 0.1 to 0.001 over the steps:
            # large steps reach the posterior from the priors, and small ones settle in it.
            if latent:
                guide = (pyro.infer.autoguide.AutoMultivariateNormal if self.guide is None else self.guide)(model)
            else:
                guide = _empty_guide
            if self.optim is None:
                optim = pyro.optim.ClippedAdam({"lr": 0.1, "lrd": 0.01 ** (1 / self.num_steps)})
            else:
                optim = copy.deepcopy(self.optim)
            svi = pyro.infer.SVI(model, guide, optim, pyro.infer.Trace_ELBO())
            losses = np.array([svi.step(*args) for _ in range(self.num_steps)])
            if not np.isfinite(losses[-1]):
                raise ValueError(
                    "SVI's loss isn't finite after the last step: lower the learning rate of optim, or check that "
                    "fn's priors are on the data's scale."
                )

            # Step 2: Draw from the guide.
            draws = pyro.infer.Predictive(guide, num_samples=self.num_samples)(*args)
        return draws, {"losses": losses}


class PyroNUTS(BasePyroEstimator):
    """MCMC with the No-U-Turn Sampler of a ``PyroAdapter``'s posterior.

    NUTS needs latent sites, and can't fit ``pyro.param`` values: give every parameter a prior with ``pyro.sample``.
    Its diagnostics are Pyro's ``MCMC.diagnostics()``: each latent site's effective sample size (``"n_eff"``) and
    split R-hat (``"r_hat"``), with the divergences and the acceptance rate.

    Parameters
    ----------
    num_samples : int, default=1000
        Number of posterior draws to keep, after the warmup.
    warmup_steps : int, optional
        Number of warmup steps, which are discarded. ``None`` for ``num_samples``, as in Pyro.
    random_state : int, numpy.random.Generator or None, optional
        Seed or generator for fitting; ``None`` draws fresh values. The global torch random state doesn't change.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyro
    >>> import pyro.distributions as dist
    >>> from pgmpy.parameterization import PyroAdapter, PyroNUTS
    >>> def unknown_mean(parents):
    ...     return dist.Normal(pyro.sample("mu", dist.Normal(0.0, 10.0)), 1.0)
    >>> y = pd.Series([1.0, 2.0, 3.0, 2.0], name="y")
    >>> node = PyroAdapter(unknown_mean, estimator=PyroNUTS(num_samples=300, random_state=0)).fit(None, y)
    >>> round(node.posterior_samples_["mu"].mean().item())
    2
    >>> sorted(node.diagnostics_["mu"])
    ['n_eff', 'r_hat']
    """

    _tags = {"name": "pyro_nuts"}

    def __init__(
        self,
        num_samples: int = 1000,
        warmup_steps: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.random_state = random_state
        super().__init__()

        _check_positive_integer("num_samples", num_samples)
        if warmup_steps is not None and (
            isinstance(warmup_steps, bool) or not isinstance(warmup_steps, Integral) or warmup_steps < 0
        ):
            raise ValueError(f"warmup_steps must be a non-negative integer or None, but is {warmup_steps!r}.")

    def estimate(
        self, model: Callable, args: tuple, latent: list[str], params: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if params:
            raise ValueError(
                f"NUTS can't fit pyro.param values, but fn has {params}: give them priors with pyro.sample, or use "
                "PyroSVI."
            )
        if not latent:
            raise ValueError("NUTS needs a latent site, a parameter with a prior from pyro.sample, but fn has none.")
        seed = int(np.random.default_rng(self.random_state).integers(2**63))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            mcmc = pyro.infer.MCMC(
                pyro.infer.NUTS(model),
                num_samples=self.num_samples,
                warmup_steps=self.warmup_steps,
                disable_progbar=not config.SHOW_PROGRESS,
            )
            mcmc.run(*args)
        return mcmc.get_samples(), mcmc.diagnostics()


def _check_positive_integer(name: str, value: Any) -> None:
    """Raise unless ``value`` is a positive integer, and not a bool."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, but is {value!r}.")


def _empty_guide(*args: Any) -> None:
    """The guide of a model without latent sites, which SVI fits by maximum likelihood."""
