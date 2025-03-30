import networkx as nx
import numpy as np
import pandas as pd
import pyro
import torch

from pgmpy import config
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.global_vars import logger
from pgmpy.models import DiscreteBayesianNetwork


class FunctionalBayesianNetwork(DiscreteBayesianNetwork):
    """
    Class for representing Functional Bayesian Network.

    Functional Bayesian Networks allow for representation of any probability
    distribution using CPDs in functional form (Functional CPD). Functional
    CPDs return a pyro.distribution object allowing for flexible representation
    of any distribution.
    """

    def __init__(self, ebunch=None, latents=set(), lavaan_str=None, dagitty_str=None):
        """
        Initializes a FunctionalBayesianNetwork.

        Parameters
        ----------
        ebunch: list
            List of edges to build the Bayesian Network. Each edge should be a tuple (u, v)
            where u, v are nodes representing the edge u -> v.

        Examples
        --------
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        """
        if config.get_backend() == "numpy":
            logger.info("Functional BN requires pytorch backend. Switching.")
            config.set_backend("torch")

        super(FunctionalBayesianNetwork, self).__init__(
            ebunch=ebunch,
            latents=latents,
            lavaan_str=lavaan_str,
            dagitty_str=dagitty_str,
        )

    def add_cpds(self, *cpds):
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
        >>> cpd2 = FunctionalCPD("x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)

        """
        for cpd in cpds:
            if not isinstance(cpd, FunctionalCPD):
                raise ValueError(
                    "Only FunctionalCPD can be added to Functional Bayesian Network."
                )

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError(f"CPD defined on variable not in the model: {cpd}")

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logger.warning(f"Replacing existing CPD for {cpd.variable}")
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):
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
        >>> cpd2 = FunctionalCPD("x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        """
        return super(FunctionalBayesianNetwork, self).get_cpds(node)

    def remove_cpds(self, *cpds):
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
        >>> cpd2 = FunctionalCPD("x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        """
        return super(FunctionalBayesianNetwork, self).remove_cpds(*cpds)

    def check_model(self):
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
                    raise ValueError(
                        f"CPD associated with {node} doesn't have proper parents associated with it."
                    )
        return True

    def simulate(self, n_samples=1000, seed=None):
        """
        Simulate samples from the model.

        Parameters
        ----------
        n_samples : int, optional (default: 1000)
            Number of samples to generate

        seed : int, optional
            The seed value for the random number generator.

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
        >>> cpd2 = FunctionalCPD("x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.simulate(n_samples=1000)
        """
        if seed is not None:
            pyro.set_rng_seed(seed)

        nodes = list(nx.topological_sort(self))
        samples = pd.DataFrame(index=range(n_samples))

        for node in nodes:
            cpd = self.get_cpds(node)
            parent_samples = samples[cpd.parents] if cpd.parents else None
            samples[node] = cpd.sample(
                n_samples=n_samples, parent_sample=parent_samples
            )

        return samples

    def fit(
        self,
        data,
        method="SVI",
        optimizer=pyro.optim.Adam({"lr": 1e-2}),
        prior_fn=None,
        num_steps=1000,
        seed=None,
        nuts_kwargs=None,
        mcmc_kwargs=None,
    ):
        """
        Fit the Bayesian network to data using Pyro's stochastic variational inference.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with observations of variables.

        method: str (default: "SVI")
            Fitting method to use. Currently supports "SVI" and "MCMC".

        optimizer: Instance of pyro optimizer (default: pyro.optim.Adam({"lr": 1e-2}))
            Only used if method is "SVI". The optimizer to use for optimization.

        prior_fn: function
            Only used if method is "MCMC". A function that returns a dictionary of
            pyro distributions for each parameter in the model.

        num_steps: int (default: 100)
            Number of optimization steps. For SVI it is the `num_steps`
            argument for pyro.infer.SVI. For MCMC, it is the `num_samples`
            argument for pyro.infer.MCMC.

        seed: int (default: None)
            Seed value for random number generator.

        nuts_kwargs: dict (default: None)
            Only used if method is "MCMC". Additional arguments to pass to
            pyro.infer.NUTS.

        mcmc_kwargs: dict (default: None)
            Only used if method is "MCMC". Additional arguments to pass to
            pyro.infer.MCMC.

        Returns
        -------
        dict: If method is "SVI", returns a dictionary of parameter values.
              If method is "MCMC", returns a dictionary of posterior samples for each parameter.

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
        ...    mu = pyro.param("x1_mu", torch.tensor(1.0))
        ...    sigma = pyro.param("x1_sigma", torch.tensor(1.0), constraint=constraints.positive)
        ...    return dist.Normal(mu, sigma)

        >>> def x2_fn(parents):
        ...    intercept = pyro.param("x2_inter", torch.tensor(1.0))
        ...    sigma = pyro.param("x2_sigma", torch.tensor(1.0), constraint=constraints.positive)
        ...    return dist.Normal(intercept + parents['x1'], sigma)

        >>> cpd1 = FunctionalCPD("x1", fn=x1_prior)
        >>> cpd2 = FunctionalCPD('x2', fn=x2_prior, parents=['x1'])
        >>> model.add_cpds(cpd1, cpd2)
        >>> params = model.fit(data, method="SVI", num_steps=100)
        >>> print(params)

        >>> def prior_fn():
        ...    return {"x1_mu": dist.Uniform(0, 1), "x1_sigma": dist.HalfNormal(5),
        ...            "x2_inter": dist.Normal(1.0), "x2_sigma": dist.HalfNormal(1)}

        >>> def x1_fn(priors, parents):
        ...    return dist.Normal(priors["x1_mu"], priors["x1_sigma"])

        >>> def x2_fn(priors, parents):
        ...    return dist.Normal(priors["x2_inter"] + parent['x1'], priors["x2_sigma"])

        >>> cpd1 = FunctionalCPD("x1", fn=x1_fn)
        >>> cpd2 = FunctionalCPD('x2', fn=x2_fn, parents=['x1'])
        >>> model.add_cpds(cpd1, cpd2)

        >>> params = model.fit(data, method="MCMC", prior_fn=prior_fn, num_steps=100)
        >>> print(params["x1_mu"].mean(), params["x1_std"].mean())
        """
        # Step 0: Checks for specified arguments.
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"data should be a pandas.DataFrame object. Got: {type(data)}."
            )

        if not isinstance(num_steps, int):
            raise ValueError(f"num_steps should be an integer. Got: {type(num_steps)}.")

        if method.lower() not in ["svi", "mcmc"]:
            raise ValueError(
                "Currently only SVI and MCMC methods are supported. method argument needs to be either 'SVI' or 'MCMC'."
            )

        # Step 1: Preprocess the data and initialize data structures.
        if seed is not None:
            pyro.set_rng_seed(seed)

        sort_nodes = list(nx.topological_sort(self))

        tensor_data = {}
        for node in sort_nodes:
            if node not in data.columns:
                raise ValueError(f"data doesn't contain column for the node: {node}.")
            else:
                tensor_data[node] = torch.tensor(
                    data[node].values,
                    dtype=config.get_dtype(),
                    device=config.get_device(),
                )

        nuts_kwargs = nuts_kwargs or {}
        mcmc_kwargs = mcmc_kwargs or {}

        cpds_dict = {node: self.get_cpds(node) for node in sort_nodes}

        # Step 2: Fit the model using the specified method.
        if method.lower() == "svi":

            def guide(tensor_data):
                pass

            # Step 2.1: Define the combined model for SVI.
            def combined_model_svi(tensor_data):
                with pyro.plate("data", data.shape[0]):
                    for node in sort_nodes:
                        pyro.sample(
                            f"{node}",
                            cpds_dict[node].fn(
                                {p: tensor_data[p] for p in cpds_dict[node].parents}
                            ),
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

        # Step 3: Fit the model using specified method
        elif method.lower() == "mcmc":
            # Step 3.1: Define the combined model for MCMC.
            def combined_model_mcmc(tensor_data):
                priors = prior_fn()
                with pyro.plate("data", data.shape[0]):
                    for node in sort_nodes:
                        pyro.sample(
                            f"{node}",
                            cpds_dict[node].fn(
                                priors,
                                {p: tensor_data[p] for p in cpds_dict[node].parents},
                            ),
                            obs=tensor_data[node],
                        )

            # Step 3.2: Fit the model using MCMC.
            nuts_kernel = pyro.infer.NUTS(combined_model_mcmc, **nuts_kwargs)
            mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_steps, **mcmc_kwargs)
            mcmc.run(tensor_data)

        # Step 4: Return the fitted parameter values.
        if method.lower() == "svi":
            return dict(pyro.get_param_store().items())
        else:
            return mcmc.get_samples()
