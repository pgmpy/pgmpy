from collections.abc import Callable, Hashable, Iterable
from typing import (
    Any,
)

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config, logger
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.models import DiscreteBayesianNetwork

pyro = _safe_import("pyro", pkg_name="pyro-ppl")
torch = _safe_import("torch")


class FunctionalBayesianNetwork(DiscreteBayesianNetwork):
    """
    Class for representing Functional Bayesian Network.

    Functional Bayesian Networks allow for flexible representation of probability distribution using CPDs in functional
    form (FunctionalCPD). As these CPDs are defined using functions that return pyro distributions, they can represent
    any distribution supported by Pyro.

    Parameters
    ----------
    ebunch : input graph, optional
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    latents : set of nodes, default=None
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, default=None
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. If None, exposures will be treated as an empty set.

    outcomes : set, optional (default: None)
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. If None, defaults to an empty set.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    # Defining a Functional Bayesian Network

    >>> from pgmpy.models import FunctionalBayesianNetwork
    >>> from pgmpy.factors.hybrid import FunctionalCPD
    >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
    >>> model.add_cpds(
    ...     FunctionalCPD("x1", lambda _: dist.Normal(0, 1)),
    ...     FunctionalCPD(
    ...         "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
    ...     ),
    ...     FunctionalCPD(
    ...         "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
    ...     ),
    ... )
    >>> model.check_model()
    True

    # Simulating data from the Functional Bayesian Network

    >>> samples = model.simulate(n_samples=1000)

    # Fitting the Functional Bayesian Network to the simulated data

    >>> fitted_params = model.fit(samples, estimator="SVI", num_steps=1000)
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        if config.get_backend() == "numpy":
            msg = (
                f"{type(self)} requires pytorch backend, currently it is "
                "set to numpy."
                "Call pgmpy.config.set_backend('torch') to switch the backend globally."
            )
            logger.info(msg)
            raise ValueError(msg)

        _check_soft_dependencies("pyro-ppl", obj=self)

        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )
        self._fit_estimator = None
        self._posterior_samples = None

    def add_cpds(self, *cpds: FunctionalCPD) -> None:
        """
        Adds FunctionalCPDs to the Bayesian Network.

        Parameters
        ----------
        cpds: instances of FunctionalCPD
            List of FunctionalCPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import pyro.distributions as dist
        >>> import numpy as np

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)

        """
        for cpd in cpds:
            if not isinstance(cpd, FunctionalCPD):
                raise ValueError("Only FunctionalCPD can be added to Functional Bayesian Network.")

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError(f"CPD defined on variable not in the model: {cpd}")

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logger.warning(f"Replacing existing CPD for {cpd.variable}")
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node: Any | None = None) -> list[FunctionalCPD] | FunctionalCPD:
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameter
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of Functional CPDs.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        """
        return super().get_cpds(node)

    def remove_cpds(self, *cpds: FunctionalCPD) -> None:
        """
        Removes the given `cpds` from the model.

        Parameters
        ----------
        *cpds: FunctionalCPD objects
            A list of FunctionalCPD objects that need to be removed from the model.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        ...

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        ...
        """
        return super().remove_cpds(*cpds)

    def check_model(self) -> bool:
        """
        Checks the model for various errors. This method checks for the following
        error -

        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass.

        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if isinstance(cpd, FunctionalCPD):
                if set(cpd.parents) != set(self.get_parents(node)):
                    raise ValueError(f"CPD associated with {node} doesn't have proper parents associated with it.")
        return True

    def simulate(
        self,
        n_samples: int = 1000,
        do: dict[Hashable, Any] | None = None,
        virtual_intervention: list[FunctionalCPD] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Simulate samples from the model.

        Parameters
        ----------
        n_samples : int, optional (default: 1000)
            Number of samples to generate

        seed : int, optional
            The seed value for the random number generator.

        do : dict, optional
            Specifies hard interventions to the model. The dict should be of
            the form {variable: value}. Incoming edges into each intervened
            variable are severed and the variable is set to the given constant
            for all rows.

        virtual_intervention : list[FunctionalCPD], optional
            A list of unconditional FunctionalCPD objects (no parents) that
            replace the corresponding node’s CPD during simulation (i.e.,
            stochastic interventions like do(X ~ Normal(...))).

        Returns
        -------
        pandas.DataFrame
            Simulated samples with columns corresponding to network variables

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.simulate(n_samples=1000)
        """
        # Step 0: Set the seed if specified, check arguments and initialize data structures.
        if seed is not None:
            pyro.set_rng_seed(seed)

        if do is None:
            do = {}

        if virtual_intervention is None:
            virtual_intervention = []

        # Check if all variables in do and virtual_intervention are valid
        extra_do = set(do.keys()) - set(self.nodes())
        if extra_do:
            raise ValueError(f"`do` contains nodes not in the model: {sorted(extra_do)}")

        vi_map = {}
        for cpd in virtual_intervention:
            if not isinstance(cpd, FunctionalCPD):
                raise ValueError("`virtual_intervention` must be a list of FunctionalCPD objects. Got {type(cpd)}")
            if cpd.variable not in set(self.nodes()):
                raise ValueError(f"Virtual intervention CPD variable not in the model: {cpd.variable}")
            if cpd.parents:
                raise ValueError(f"Virtual intervention CPD for {cpd.variable} must be unconditional (no parents).")
            vi_map[cpd.variable] = cpd

        overlap = set(do.keys()) & set(vi_map.keys())
        if overlap:
            raise ValueError(
                f"Cannot specify both `do` and `virtual_intervention` for the same node(s): {sorted(overlap)}"
            )

        nodes = list(nx.topological_sort(self))
        samples = pd.DataFrame(index=range(n_samples))

        # Step 1: Simulate data
        for node in nodes:
            # Step 1.1: Handle hard interventions
            if node in do:
                samples[node] = np.full(n_samples, do[node])
                continue

            # Step 1.2: Handle virtual interventions
            if node in vi_map:
                samples[node] = vi_map[node].sample(n_samples=n_samples, parent_sample=None)
                continue

            # Step 1.3: Standard sampling from the node's CPD
            cpd = self.get_cpds(node)
            parent_samples = samples[cpd.parents] if cpd.parents else None
            samples[node] = cpd.sample(n_samples=n_samples, parent_sample=parent_samples)

        # Step 2: Return the simulated samples
        return samples

    def fit(
        self,
        data: pd.DataFrame,
        estimator: str = "SVI",
        optimizer: pyro.optim.PyroOptim = pyro.optim.Adam({"lr": 1e-2}),
        prior_fn: Callable | None = None,
        num_steps: int = 1000,
        seed: int | None = None,
        nuts_kwargs: dict | None = None,
        mcmc_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """
        Fit the Bayesian network to data using Pyro's stochastic variational inference.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with observations of variables.

        estimator: str (default: "SVI")
            Fitting method to use. Currently supports "SVI" and "MCMC".

        optimizer: Instance of pyro optimizer (default: pyro.optim.Adam({"lr": 1e-2}))
            Only used if `estimator` is "SVI". The optimizer to use for optimization.

        prior_fn: function
            Only used if `estimator` is "MCMC". A function that returns a dictionary of
            pyro distributions for each parameter in the model.

        num_steps: int (default: 100)
            Number of optimization steps. For SVI it is the `num_steps`
            argument for pyro.infer.SVI. For MCMC, it is the `num_samples`
            argument for pyro.infer.MCMC.

        seed: int (default: None)
            Seed value for random number generator.

        nuts_kwargs: dict (default: None)
            Only used if `estimator` is "MCMC". Additional arguments to pass to
            pyro.infer.NUTS.

        mcmc_kwargs: dict (default: None)
            Only used if `estimator` is "MCMC". Additional arguments to pass to
            pyro.infer.MCMC.

        Returns
        -------
        dict: If `estimator` is "SVI", returns a dictionary of parameter values.
              If `estimator` is "MCMC", returns a dictionary of posterior samples for each parameter.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2")])
        >>> x1 = np.random.normal(0.2, 0.8, size=10000)
        >>> x2 = np.random.normal(0.6 + x1, 1)
        >>> data = pd.DataFrame({"x1": x1, "x2": x2})

        >>> def x1_fn(parents):
        ...     mu = pyro.param("x1_mu", torch.tensor(1.0))
        ...     sigma = pyro.param(
        ...         "x1_sigma", torch.tensor(1.0), constraint=constraints.positive
        ...     )
        ...     return dist.Normal(mu, sigma)
        ...

        >>> def x2_fn(parents):
        ...     intercept = pyro.param("x2_inter", torch.tensor(1.0))
        ...     sigma = pyro.param(
        ...         "x2_sigma", torch.tensor(1.0), constraint=constraints.positive
        ...     )
        ...     return dist.Normal(intercept + parents["x1"], sigma)
        ...

        >>> cpd1 = FunctionalCPD("x1", fn=x1_prior)
        >>> cpd2 = FunctionalCPD("x2", fn=x2_prior, parents=["x1"])
        >>> model.add_cpds(cpd1, cpd2)
        >>> params = model.fit(data, estimator="SVI", num_steps=100)
        >>> print(params)

        >>> def prior_fn():
        ...     return {
        ...         "x1_mu": dist.Uniform(0, 1),
        ...         "x1_sigma": dist.HalfNormal(5),
        ...         "x2_inter": dist.Normal(1.0),
        ...         "x2_sigma": dist.HalfNormal(1),
        ...     }
        ...

        >>> def x1_fn(priors, parents):
        ...     return dist.Normal(priors["x1_mu"], priors["x1_sigma"])
        ...

        >>> def x2_fn(priors, parents):
        ...     return dist.Normal(
        ...         priors["x2_inter"] + parent["x1"], priors["x2_sigma"]
        ...     )
        ...

        >>> cpd1 = FunctionalCPD("x1", fn=x1_fn)
        >>> cpd2 = FunctionalCPD("x2", fn=x2_fn, parents=["x1"])
        >>> model.add_cpds(cpd1, cpd2)

        >>> params = model.fit(data, estimator="MCMC", prior_fn=prior_fn, num_steps=100)
        >>> print(params["x1_mu"].mean(), params["x1_std"].mean())
        """
        # Step 0: Checks for specified arguments.
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data should be a pandas.DataFrame object. Got: {type(data)}.")

        if not isinstance(num_steps, int):
            raise ValueError(f"num_steps should be an integer. Got: {type(num_steps)}.")

        if estimator.lower() not in ["svi", "mcmc"]:
            raise ValueError(f"`estimator` argument needs to be either 'SVI' or 'MCMC'. Got: {estimator}.")

        # Step 1: Preprocess the data and initialize data structures.
        if seed is not None:
            pyro.set_rng_seed(seed)

        sort_nodes = list(nx.topological_sort(self))

        tensor_data = {}
        for node in sort_nodes:
            if node not in data.columns:
                raise ValueError(f"data doesn't contain column for the node: {node}.")
            else:
                import torch

                tensor_data[node] = torch.tensor(
                    data[node].values,
                    dtype=config.get_dtype(),
                    device=config.get_device(),
                )

        nuts_kwargs = nuts_kwargs or {}
        mcmc_kwargs = mcmc_kwargs or {}

        cpds_dict = {node: self.get_cpds(node) for node in sort_nodes}
        self._fit_estimator = estimator.lower()
        self._posterior_samples = None

        # Step 2: Fit the model using the specified method.
        if estimator.lower() == "svi":

            def guide(tensor_data):
                pass

            # Step 2.1: Define the combined model for SVI.
            def combined_model_svi(tensor_data):
                with pyro.plate("data", data.shape[0]):
                    for node in sort_nodes:
                        pyro.sample(
                            f"{node}",
                            cpds_dict[node].fn({p: tensor_data[p] for p in cpds_dict[node].parents}),
                            obs=tensor_data[node],
                        )

            # Step 2.2: Fit the model using SVI.
            svi = pyro.infer.SVI(
                model=combined_model_svi,
                guide=guide,
                optim=optimizer,
                loss=pyro.infer.Trace_ELBO(),
            )

            for step in range(num_steps):
                loss = svi.step(tensor_data)
                if step % 50 == 0:
                    logger.info(f"Step {step} | Loss: {loss:.4f}")

        # Step 3: Fit the model using specified estimator
        elif estimator.lower() == "mcmc":
            # Step 3.1: Define the combined model for MCMC.
            def combined_model_mcmc(tensor_data):
                priors_dists = prior_fn()
                priors_vals = {name: pyro.sample(name, d) for name, d in priors_dists.items()}

                with pyro.plate("data", data.shape[0]):
                    for node in sort_nodes:
                        dist_node = cpds_dict[node].fn(
                            priors_vals,
                            {p: tensor_data[p] for p in cpds_dict[node].parents},
                        )
                        pyro.sample(f"{node}", dist_node, obs=tensor_data[node])

            # Step 3.2: Fit the model using MCMC.
            nuts_kernel = pyro.infer.NUTS(combined_model_mcmc, **nuts_kwargs)
            mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_steps, **mcmc_kwargs)
            mcmc.run(tensor_data)

        # Step 4: Return the fitted parameter values.
        if estimator.lower() == "svi":
            return dict(pyro.get_param_store().items())
        else:
            self._posterior_samples = mcmc.get_samples()
            return self._posterior_samples

    def _validate_prediction_data(self, data: pd.DataFrame, allow_nan: bool) -> list[Hashable]:
        """
        Validate inputs for prediction methods and return the model's topological order.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data should be a pandas.DataFrame object. Got: {type(data)}.")

        extra_columns = set(data.columns) - set(self.nodes())
        if extra_columns:
            raise ValueError("Data has variables which are not in the model")

        if self._fit_estimator == "mcmc":
            raise NotImplementedError(
                "Prediction after fitting FunctionalBayesianNetwork with estimator='MCMC' is not yet supported."
            )

        self.check_model()
        topo_order = list(nx.topological_sort(self))

        if allow_nan:
            if set(data.columns) == set(self.nodes()) and not data.isna().any().any():
                raise ValueError("No variable missing in data. Nothing to predict")
        else:
            if data.isna().any().any():
                raise ValueError(
                    "`predict_probability` does not support NaN values. "
                    "Drop columns corresponding to variables to predict."
                )

            missing_variables = [var for var in topo_order if var not in data.columns]
            if len(missing_variables) == 0:
                raise ValueError("No variable missing in data. Nothing to predict")

        return topo_order

    def _to_prediction_tensor(self, value: Any) -> torch.Tensor:
        """
        Convert a Python or NumPy value into a torch tensor compatible with the configured backend.
        """
        tensor = torch.as_tensor(value, device=config.get_device())
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=config.get_dtype())
        return tensor

    def _to_numpy_samples(self, values: Any, node: Hashable, n_samples: int) -> np.ndarray:
        """
        Convert vectorized samples into a 1D numpy array with `n_samples` elements.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        values = np.asarray(values)
        values = np.squeeze(values)

        if values.ndim == 0:
            values = np.repeat(values.item(), n_samples)

        if values.ndim != 1 or values.shape[0] != n_samples:
            raise ValueError(
                f"Vectorized CPD for {node} must return exactly {n_samples} scalar samples. "
                f"Got array with shape {values.shape}."
            )

        return values

    def _to_log_prob_vector(self, log_prob: torch.Tensor, node: Hashable, n_samples: int) -> torch.Tensor:
        """
        Convert a batched log-probability tensor into a 1D tensor of per-sample log-probabilities.
        """
        if not isinstance(log_prob, torch.Tensor):
            log_prob = self._to_prediction_tensor(log_prob)

        if log_prob.ndim == 0:
            log_prob = log_prob.repeat(n_samples)
        elif log_prob.shape[0] != n_samples:
            raise ValueError(
                f"Vectorized CPD for {node} returned log_prob with incompatible shape {tuple(log_prob.shape)}."
            )
        else:
            log_prob = log_prob.reshape(n_samples, -1).sum(dim=1)

        return log_prob.to(dtype=config.get_dtype(), device=config.get_device())

    def _normalize_log_weights(self, log_weights: torch.Tensor) -> np.ndarray:
        """
        Normalize log-weights using a stable log-sum-exp transform.
        """
        finite_mask = torch.isfinite(log_weights)
        if not torch.any(finite_mask):
            raise ValueError("Evidence has zero probability under the model.")

        finite_weights = log_weights[finite_mask]
        shifted = torch.exp(finite_weights - torch.max(finite_weights))
        weight_sum = shifted.sum()
        if weight_sum <= 0:
            raise ValueError("Evidence has zero probability under the model.")

        weights = torch.zeros_like(log_weights)
        weights[finite_mask] = shifted / weight_sum
        return weights.detach().cpu().numpy()

    def _get_distribution_descriptor(self, distribution: Any) -> dict[str, Any]:
        """
        Classify a Pyro distribution for prediction summaries.
        """
        support = getattr(distribution, "support", None)
        is_discrete = bool(getattr(support, "is_discrete", False))

        if getattr(distribution, "has_enumerate_support", False) and is_discrete:
            try:
                states = distribution.enumerate_support(expand=False)
            except TypeError:
                states = distribution.enumerate_support()

            if isinstance(states, torch.Tensor):
                states = states.detach().cpu().numpy()

            states = np.asarray(states)
            if states.ndim > 1:
                states = states[:, 0]

            return {
                "kind": "finite_discrete",
                "states": [state.item() if np.asarray(state).ndim == 0 else state for state in states],
            }

        if is_discrete:
            return {"kind": "discrete"}

        return {"kind": "continuous"}

    def _weighted_mode(self, values: np.ndarray, weights: np.ndarray, states: list[Any] | None = None) -> Any:
        """
        Compute a weighted mode from posterior samples.
        """
        candidate_states = states if states is not None else list(np.unique(values))
        state_probs = [weights[values == state].sum() for state in candidate_states]
        return candidate_states[int(np.argmax(state_probs))]

    def _weighted_mean_and_cov(self, samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted posterior means and covariances from samples.
        """
        mean = weights @ samples
        centered = samples - mean
        cov = centered.T @ (centered * weights[:, None])
        return mean, cov

    def _posterior_importance_samples(
        self,
        evidence: dict[Hashable, Any],
        missing_vars: list[Hashable],
        topo_order: list[Hashable],
        cpds_dict: dict[Hashable, FunctionalCPD],
        n_samples: int,
        seed: int | None = None,
    ) -> tuple[dict[Hashable, np.ndarray], np.ndarray, dict[Hashable, dict[str, Any]]]:
        """
        Approximate posterior samples for missing variables using ancestral importance sampling.
        """
        if seed is not None:
            pyro.set_rng_seed(seed)

        log_weights = torch.zeros(n_samples, dtype=config.get_dtype(), device=config.get_device())
        particles: dict[Hashable, np.ndarray] = {}
        descriptors: dict[Hashable, dict[str, Any]] = {}
        missing_set = set(missing_vars)

        with torch.no_grad():
            for node in topo_order:
                cpd = cpds_dict[node]

                if cpd.vectorized:
                    parent_sample = None
                    if cpd.parents:
                        parent_sample = pd.DataFrame({parent: particles[parent] for parent in cpd.parents})

                    distribution = cpd.fn(parent_sample)

                    if node in missing_set:
                        descriptors[node] = self._get_distribution_descriptor(distribution)
                        particles[node] = self._to_numpy_samples(distribution.sample(), node=node, n_samples=n_samples)
                    else:
                        obs_tensor = self._to_prediction_tensor(np.repeat(evidence[node], n_samples))
                        log_weights += self._to_log_prob_vector(
                            distribution.log_prob(obs_tensor),
                            node=node,
                            n_samples=n_samples,
                        )
                        particles[node] = np.repeat(evidence[node], n_samples)

                else:
                    node_particles = []
                    for particle_index in range(n_samples):
                        if cpd.parents:
                            parent_values = {
                                parent: self._to_prediction_tensor(particles[parent][particle_index])
                                for parent in cpd.parents
                            }
                        else:
                            parent_values = None

                        distribution = cpd.fn(parent_values)

                        if node in missing_set:
                            descriptors.setdefault(node, self._get_distribution_descriptor(distribution))
                            sample = distribution.sample()
                            if isinstance(sample, torch.Tensor):
                                sample = sample.detach().cpu().numpy()
                            sample = np.asarray(sample).item()
                            node_particles.append(sample)
                        else:
                            obs_value = self._to_prediction_tensor(evidence[node])
                            log_prob = distribution.log_prob(obs_value)
                            if isinstance(log_prob, torch.Tensor) and log_prob.ndim > 0:
                                log_prob = log_prob.sum()
                            log_weights[particle_index] += log_prob
                            node_particles.append(evidence[node])

                    particles[node] = np.asarray(node_particles)

        weights = self._normalize_log_weights(log_weights)
        posterior_samples = {var: particles[var] for var in missing_vars}
        posterior_descriptors = {var: descriptors[var] for var in missing_vars}
        return posterior_samples, weights, posterior_descriptors

    def predict(
        self,
        data: pd.DataFrame,
        stochastic: bool = False,
        n_samples: int = 1000,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Predict missing variables from a Functional Bayesian Network using approximate posterior inference.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing observed variables. Missing variables can either be omitted
            as columns or indicated with NaN values.

        stochastic : bool, default=False
            If True, return a joint sample from the approximate posterior of the missing variables.
            If False, return posterior means for continuous variables and posterior modes for
            discrete variables.

        n_samples : int, default=1000
            Number of importance samples used to approximate the posterior.

        seed : int, optional
            Random seed for posterior approximation and stochastic prediction.

        Returns
        -------
        pandas.DataFrame
            A completed DataFrame with predictions for all missing values.
        """
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples should be a positive integer. Got: {n_samples}.")

        topo_order = self._validate_prediction_data(data=data, allow_nan=True)
        cpds_dict = {node: self.get_cpds(node) for node in topo_order}
        all_columns = data.columns.tolist() + [node for node in topo_order if node not in data.columns]
        rng = np.random.default_rng(seed)
        predictions = []

        for index, data_point in data.iterrows():
            missing_vars = [
                node for node in topo_order if node not in data.columns or pd.isna(data_point.get(node, np.nan))
            ]
            completed_row = data_point.to_dict()

            if missing_vars:
                evidence = {
                    node: data_point[node] for node in topo_order if node in data.columns and pd.notna(data_point[node])
                }
                row_seed = None if seed is None else int(rng.integers(0, np.iinfo(np.int32).max))
                samples, weights, descriptors = self._posterior_importance_samples(
                    evidence=evidence,
                    missing_vars=missing_vars,
                    topo_order=topo_order,
                    cpds_dict=cpds_dict,
                    n_samples=n_samples,
                    seed=row_seed,
                )

                if stochastic:
                    particle_index = int(rng.choice(n_samples, p=weights))
                    for node in missing_vars:
                        completed_row[node] = samples[node][particle_index]
                else:
                    for node in missing_vars:
                        kind = descriptors[node]["kind"]
                        if kind == "continuous":
                            completed_row[node] = float(np.dot(weights, samples[node]))
                        elif kind == "finite_discrete":
                            completed_row[node] = self._weighted_mode(
                                values=samples[node],
                                weights=weights,
                                states=descriptors[node]["states"],
                            )
                        else:
                            completed_row[node] = self._weighted_mode(values=samples[node], weights=weights)

            predictions.append(pd.Series(completed_row, name=index))

        return pd.DataFrame(predictions).reindex(columns=all_columns)

    def predict_probability(
        self,
        data: pd.DataFrame,
        n_samples: int = 1000,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | tuple[list[str], np.ndarray, np.ndarray]:
        """
        Predict posterior distributions for missing variables using approximate posterior inference.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing observed variables. Variables to predict must be omitted as columns.

        n_samples : int, default=1000
            Number of importance samples used to approximate the posterior.

        seed : int, optional
            Random seed for posterior approximation.

        Returns
        -------
        pandas.DataFrame or tuple[list[str], np.ndarray, np.ndarray]
            If all missing variables have finite discrete support, returns a DataFrame with one
            probability column per state. If all missing variables are continuous, returns a tuple
            of `(variables, mean, covariance)` where `mean` has shape `(n_rows, n_missing)` and
            `covariance` has shape `(n_rows, n_missing, n_missing)`.
        """
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples should be a positive integer. Got: {n_samples}.")

        topo_order = self._validate_prediction_data(data=data, allow_nan=False)
        cpds_dict = {node: self.get_cpds(node) for node in topo_order}
        missing_vars = [node for node in topo_order if node not in data.columns]
        rng = np.random.default_rng(seed)

        posterior_results = []
        kinds = set()
        reference_descriptors = None

        for _, data_point in data.iterrows():
            evidence = {node: data_point[node] for node in data.columns}
            row_seed = None if seed is None else int(rng.integers(0, np.iinfo(np.int32).max))
            samples, weights, descriptors = self._posterior_importance_samples(
                evidence=evidence,
                missing_vars=missing_vars,
                topo_order=topo_order,
                cpds_dict=cpds_dict,
                n_samples=n_samples,
                seed=row_seed,
            )

            posterior_results.append((samples, weights, descriptors))
            kinds.update(descriptor["kind"] for descriptor in descriptors.values())
            if reference_descriptors is None:
                reference_descriptors = descriptors

        if kinds == {"finite_discrete"}:
            pred_values = {
                f"{var}_{state}": [] for var in missing_vars for state in reference_descriptors[var]["states"]
            }

            for samples, weights, descriptors in posterior_results:
                for var in missing_vars:
                    for state in descriptors[var]["states"]:
                        pred_values[f"{var}_{state}"].append(weights[samples[var] == state].sum())

            return pd.DataFrame(pred_values, index=data.index)

        if kinds == {"continuous"}:
            means = []
            covariances = []

            for samples, weights, _ in posterior_results:
                sample_matrix = np.column_stack([samples[var] for var in missing_vars])
                mean, covariance = self._weighted_mean_and_cov(sample_matrix, weights)
                means.append(mean)
                covariances.append(covariance)

            return missing_vars, np.vstack(means), np.stack(covariances)

        raise NotImplementedError(
            "`predict_probability` currently supports either all-continuous missing variables "
            "or all finite-support discrete missing variables."
        )
