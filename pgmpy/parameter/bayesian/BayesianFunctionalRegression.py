from typing import Literal

from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.distributions.converter.PyroToSkpro import PyroToSkpro
from pgmpy.parameter._base import BaseParameter

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


def to_tensor(arr, device, reshape=False):
    """
    helper function

    pd.DataFrame -> torch.tensor

    FYI: `pgmpy.utils.compat_fns` only support tensor <-> numpy

    """
    tensor = torch.as_tensor(
        arr.to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    if reshape:
        tensor = tensor.reshape(-1)

    return tensor


class BayesianFunctionalRegression(BaseParameter):
    """Base class for parameter classes using `pyro` in pgmpy.

    Parameters
    ----------
    model : callable
        Pyro probabilistic model used for inference. The callable must accept the
        predictor tensor as its first argument and may accept the response tensor
        as its second argument.

    estimator : {"svi", "mcmc"}, default="svi"
        Inference algorithm used to estimate the posterior distribution.
        `"svi"` uses stochastic variational inference, while `"mcmc"` uses
        Markov chain Monte Carlo sampling.

    num_samples : int, default=1000
        Number of posterior predictive samples to draw. When `estimator="mcmc"`,
        this is also the number of posterior samples generated during fitting.

    posterior : str, default="normal"
        Posterior distribution specification passed to
        :class:`PyroToSkpro` for converting Pyro samples into an skpro
        distribution.

    return_sites : str, default="obs"
        Name of the Pyro sample site whose posterior predictive samples are
        returned.

    device : str or torch.device, default="cpu"
        Device on which input tensors and Pyro computations are placed, such as
        `"cpu"` or `"cuda"`.

    warmup_steps : int, default=200
        Number of MCMC warm-up steps. Used only when `estimator="mcmc"`.

    guide : callable, optional
        Pyro guide used for variational inference. When `None`,
        :class:`pyro.infer.autoguide.AutoNormal` is constructed from `model`.
        Used only when `estimator="svi"`.

    optim : pyro.optim.PyroOptim, optional
        Pyro optimizer used for variational inference. By default,
        :class:`pyro.optim.Adam` is used with a learning rate of `0.3`.
        Used only when `estimator="svi"`.

    loss : pyro.infer.ELBO, optional
        Loss function used for variational inference. By default,
        :class:`pyro.infer.Trace_ELBO` is used. Used only when
        `estimator="svi"`.

    num_iterations : int, default=1500
        Number of optimization iterations performed during stochastic
        variational inference. Used only when `estimator="svi"`.

    svi_log : bool, default=False
        Whether to print the average SVI loss every 100 iterations. Used only when
        `estimator="svi"`.

    Example
    -------
    >>> from pgmpy.parameter.bayesian.BayesianFunctionalRegression import BayesianFunctionalRegression
    >>> from skbase.utils.dependencies import _safe_import
    >>> import numpy as np
    >>> import pandas as pd
    >>> torch = _safe_import("torch")
    >>> pyro = _safe_import("pyro", pkg_name="pyro-ppl")
    >>> rng = np.random.default_rng(42)
    >>> n_samples = 1000

    >>> A = rng.normal(
    ...     loc=0.0,
    ...     scale=1.0,
    ...     size=n_samples,
    ... )

    >>> B = (
    ...     A + rng.normal(
    ...         loc=1.0,
    ...         scale=0.5,
    ...         size=n_samples,
    ...     )
    ... )

    >>> X = pd.DataFrame(
    ...     {
    ...         "A": A,
    ...     }
    ... )
    >>> y = pd.DataFrame(
    ...     {
    ...         "B": B,
    ...     }
    ... )

    >>> def model(X_tensor, y_tensor=None):
    ...     intercept = pyro.sample("intercept", pyro.distributions.Normal(0.0, 10.0))
    ...     coeff = pyro.sample("coeff", pyro.distributions.Normal(0.0, 1.0))
    ...     sigma = pyro.sample("sigma", pyro.distributions.Uniform(0.0, 10.0))
    ...     mean = intercept + coeff * X_tensor[:, 0]
    ...     with pyro.plate("data", len(X_tensor[:, 0])):
    ...         pyro.sample("obs", pyro.distributions.Normal(mean, sigma), obs=y_tensor)

    >>> mcmc_regressor = BayesianFunctionalRegression(
    ...     model=model,
    ...     estimator="mcmc",
    ...     warmup_steps=10,
    ...     num_samples=100,
    ... )
    >>> mcmc_regressor.fit(X, y) # doctest: +SKIP
    BayesianFunctionalRegression()
    >>> normal = mcmc_regressor.predict_proba(X[:5]) # doctest: +SKIP
    >>> normal # doctest: +SKIP
    Normal()

    >>> svi_regressor = BayesianFunctionalRegression(
    ...     model=model,
    ...     estimator="svi",
    ...     guide=pyro.infer.autoguide.AutoNormal(model),
    ...     optim=pyro.optim.Adam({"lr": 0.3}),
    ...     loss=pyro.infer.Trace_ELBO(),
    ...     num_iterations=500,
    ...     svi_log=True,
    ... )
    >>> svi_regressor.fit(X, y) # doctest: +SKIP
    BayesianFunctionalRegression()
    >>> normal = svi_regressor.predict_proba(X[:5]) # doctest: +SKIP
    >>> normal # doctest: +SKIP
    Normal()

    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "can_be_root": False,
        "python_dependencies": ("pyro-ppl"),
    }

    def __init__(
        self,
        model,
        estimator: Literal["svi", "mcmc"] = "svi",
        num_samples=1000,
        *,
        posterior="normal",
        return_sites="obs",
        device="cpu",
        # MCMC
        warmup_steps=200,
        # SVI
        guide=None,
        optim=pyro.optim.Adam({"lr": 0.3}),
        loss=pyro.infer.Trace_ELBO(),
        num_iterations=1500,
        svi_log=False,
    ):
        super().__init__()
        _check_soft_dependencies(self.get_tag("python_dependencies"))

        self.model = model
        self.estimator = estimator
        self.num_samples = num_samples

        self.posterior = posterior
        self.return_sites = return_sites
        self.device = device

        self.warmup_steps = warmup_steps

        self.guide = guide
        self.optim = optim
        self.loss = loss
        self.num_iterations = num_iterations
        self.svi_log = svi_log

    def _fit(self, X, y=None, sample_weight=None):
        X_tensor = to_tensor(X, self.device)
        y_tensor = to_tensor(y, self.device, reshape=True)

        pyro.clear_param_store()
        if self.estimator == "mcmc":
            self.warmup_steps_ = self.warmup_steps

            nuts_kernel = pyro.infer.NUTS(self.model)
            self.mcmc_ = pyro.infer.MCMC(nuts_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps_)
            self.mcmc_.run(X_tensor, y_tensor)

        elif self.estimator == "svi":
            self.guide_ = self.guide if self.guide is not None else pyro.infer.autoguide.AutoNormal(self.model)
            self.optim_ = self.optim
            self.loss_ = self.loss

            svi = pyro.infer.SVI(
                model=self.model,
                guide=self.guide_,
                optim=self.optim_,
                loss=self.loss_,
            )

            for j in range(self.num_iterations):
                loss = svi.step(X_tensor, y_tensor)

                if self.svi_log and j % 100 == 0:
                    print(f"[iteration {j + 1:04d}] loss: {loss / X_tensor.shape[0]:.4f}")

        self.posterior_ = PyroToSkpro(self.posterior)
        self.columns_ = y.columns[0]
        if self.device == "cpu":
            self.parallel_ = False
        else:
            self.parallel_ = True
        return self

    def _predict_proba(self, X):
        index = X.index
        X_tensor = to_tensor(X, self.device)

        if self.estimator == "mcmc":
            predictive = pyro.infer.Predictive(
                model=self.model,
                posterior_samples=self.mcmc_.get_samples(),
                num_samples=self.num_samples,
                return_sites=(self.return_sites,),
                parallel=self.parallel_,
            )
        elif self.estimator == "svi":
            predictive = pyro.infer.Predictive(
                model=self.model,
                guide=self.guide_,
                num_samples=self.num_samples,
                return_sites=(self.return_sites,),
                parallel=self.parallel_,
            )

        with torch.no_grad():
            samples = predictive(X_tensor)

        SkproDistribution = self.posterior_.convert(samples, index, [self.columns_], self.return_sites)
        return SkproDistribution
