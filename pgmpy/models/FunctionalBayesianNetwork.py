import networkx as nx
import numpy as np
import pandas as pd
import pyro
import torch
import torch.distributions.constraints as constraints

from pgmpy import config
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.global_vars import logger
from pgmpy.models import BayesianNetwork


class FunctionalBayesianNetwork(BayesianNetwork):
    """
    Class for representing Functional Bayesian Network.

    Functional Bayesian Networks allow for representation of any probability
    distribution using CPDs in functional form (Functional CPD). Functional
    CPDs return a pyro.distribution object allowing for flexible representation
    of any distribution.
    """

    def __init__(self, ebunch=None):
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

        super(FunctionalBayesianNetwork, self).__init__(ebunch)

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
                raise ValueError("Only FunctionalCPD can be added.")

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

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
                        "CPD associated with %s doesn't have "
                        "proper parents associated with it." % node
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
        learning_rate=1e-2,
        num_steps=1000,
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

        learning_rate: float (default: 1e-2)
            Learning rate to use for the fitting.

        num_steps: int (default: 100)
            Number of optimization steps. For SVI it is the `num_steps`
            argument for pyro.infer.SVI. For MCMC, it is the `num_samples`
            argument for pyro.infer.MCMC.

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
        >>> x1 = np.random.normal(1, 2, size=10000)
        >>> x2 = np.random.normal(5 + x1, 1)
        >>> data = pd.DataFrame({"x1": x1, "x2": x2})
        >>> def x1_prior():
        ...    mu = pyro.sample("x1_mu", dist.Normal(0, 10))
        ...    sigma = pyro.sample("x1_sigma", dist.HalfNormal(5))
        ...    return dist.Normal(mu, sigma)
        >>> def x2_prior(parent):
        ...    mu = pyro.param("x2_mu", torch.tensor(1.0)) + parent["x1"]
        ...    sigma = positive_param("x2_sigma", 1.0)
        ...    return dist.Normal(mu, sigma)

        >>> cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        >>> cpd2 = FunctionalCPD('x2', fn=lambda parent: x2_prior(parent), parents=['x1'])
        >>> model.add_cpds(cpd1, cpd2)
        >>> params = model.fit(data, method="SVI", learning_rate=0.05, num_steps=100)
        >>> print(params)

        >>> def x1_prior():
        ...    mu = pyro.sample("x1_mu", dist.Normal(0, 10))
        ...    sigma = pyro.sample("x1_sigma", dist.HalfNormal(5))
        ...    return dist.Normal(mu, sigma)

        >>> def x2_prior(parent):
        ...    mu = pyro.sample("x2_mu", dist.Normal(5, 1))
        ...    sigma = pyro.sample("x2_sigma", dist.HalfNormal(2))
        ...    return dist.Normal(mu + parent['x1'], sigma)

        >>> cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        >>> cpd2 = FunctionalCPD('x2', fn=lambda parent: x2_prior(parent), parents=['x1'])

        >>> params = model.fit(data, method="MCMC", num_steps=100, mcmc_kwargs={"mp_context": "fork"})
        >>> print(params["x1_mu"].mean(), params["x1_std"].mean())
        """
        # Step 0: Checks for specified arguments.
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Specify data as a pandas Dataframe.")

        if not isinstance(learning_rate, float):
            raise ValueError(
                f"Learning rate should be float type, not {type(learning_rate)}"
            )

        if not isinstance(num_steps, int):
            raise ValueError(
                f"Number of steps should be int type, not {type(num_steps)}"
            )

        if method.lower() not in ["svi", "mcmc"]:
            raise ValueError("Current implementation only support SVI or MCMC.")

        # Step 1: Preprocess the data and initialize data structures.
        sort_nodes = list(nx.topological_sort(self))

        tensor_data = {}
        for node in sort_nodes:
            if node not in data.columns:
                raise ValueError(f"Observation not found for variable {node}")
            else:
                tensor_data[node] = torch.tensor(
                    data[node].values, dtype=config.get_dtype()
                ).to(config.get_device())

        nuts_kwargs = nuts_kwargs or {}
        mcmc_kwargs = mcmc_kwargs or {}

        # No latent variables to approximate
        def guide(tensor_data):
            pass

        optimizer = pyro.optim.Adam({"lr": learning_rate})

        if method.lower() == "mcmc":
            params = {}

        # Step 2: Define a full pyro model using the CPDs.
        cpds_dict = {node: self.get_cpds(node) for node in sort_nodes}

        def combined_model(tensor_data):
            with pyro.plate("data", data.shape[0]):
                for node in sort_nodes:
                    import ipdb

                    ipdb.set_trace()
                    pyro.sample(
                        f"{node}",
                        cpds_dict[node].fn(
                            {p: tensor_data[p] for p in cpds_dict[node].parents}
                        ),
                        obs=tensor_data[node],
                    )

        import ipdb

        ipdb.set_trace()
        # Step 3: Fit the model using the specified method.
        if method.lower() == "svi":
            svi = pyro.infer.SVI(
                model=combined_model,
                guide=guide,
                optim=optimizer,
                loss=pyro.infer.Trace_ELBO(),
            )

            for step in range(num_steps):
                loss = svi.step(tensor_data)
                if step % 50 == 0:
                    logger.info(f"Step {step} | Loss: {loss:.4f}")

        else:
            nuts_kernel = pyro.infer.NUTS(combined_model, **nuts_kwargs)
            mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_steps, **mcmc_kwargs)
            mcmc.run()
            samples = mcmc.get_samples()
            params.update(samples)

        # Step 4: Return the fitted parameter values.
        if method.lower() == "svi":
            params = pyro.get_param_store()
            return {name: params[name] for name in params.keys()}
        else:
            return params
