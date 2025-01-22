import networkx as nx
import numpy as np
import pandas as pd
import pyro
import torch
import torch.distributions.constraints as constraints

from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.global_vars import logger
from pgmpy.models import BayesianNetwork


class FunctionalBayesianNetwork(BayesianNetwork):
    """
    A Functional Gaussian Bayesian Network is a Bayesian Network,
    whose variables can be discrete or continuous, and where all of the CPDs
    are defined by FunctionalCPD.

    An important result is that the Functional Bayesian Networks
    provide flexible representation for the class of multiples uni/multi-variate
    distributions.
    """

    def add_cpds(self, *cpds):
        """
        Add Functional CPD (Conditional Probability Distribution)
        to the Bayesian Network.

        Parameters
        ----------
        cpds  :  instances of FunctionalCPD
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
                    continue
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
        Removes the cpds that are provided in the argument.

        Parameters
        ----------

        *cpds: FunctionalCPD object
            A FunctionalCPD object on any subset of the variables
            of the model which is to be associated with the model.

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
        Simulate samples from a FunctionalBayesianNetwork.

        Parameters
        ----------
        n_samples : int, optional (default=1000)
            Number of samples to generate

        seed : int, optional
            Random seed for reproducibility

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
            import pyro

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
        guides=None,
        method="SVI",
        learning_rate=1e-4,
        num_steps=100,
        nuts_kwargs=None,
        mcmc_kwargs=None,
    ):
        """
        Fit the Bayesian network to data using Pyro's stochastic variational inference.

        Parameters:
            data (pd.DataFrame) : DataFrame with observations of variables.
            guides (dict) : Dictionary of Pyro guide functions for variational inference.
            method (str) : Approximation methods for posterior distribution.
            learning_rate (float) : Learning rate for optimization
            num_steps (int) : Number of optimization steps for each variable.

        Returns:
            dict: Samples for each register parameter.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> x1 = np.random.normal(1, 2, size=10000)
        >>> data = pd.DataFrame({"x1": x1})
        >>> def x1_prior():
        ...    mu = pyro.sample("x1_mu", dist.Normal(0, 10))
        ...    sigma = pyro.sample("x1_sigma", dist.HalfNormal(5))
        ...    return dist.Normal(mu, sigma)

        >>> cpd1 = FunctionalCPD("x1", lambda: x1_prior())
        >>> model.add_cpds(cpd1)
        >>> params = model.fit(data, method="MCMC", num_steps=100)
        >>> print(params["x1_mu"].mean(), params["x1_std"].mean())
        """

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
        if method not in ["SVI", "MCMC"]:
            raise ValueError("Current implementation only support SVI or MCMC.")

        sort_nodes = list(nx.topological_sort(self))

        if method == "SVI":
            if guides is None:
                raise ValueError("Specify guides function for SVI.")
            else:
                if not isinstance(guides, dict):
                    raise ValueError(
                        f"Specify guides as dictionary, not {type(guides)}"
                    )
                for variable in data.columns:
                    if variable not in guides.keys():
                        raise ValueError(f"Guide not found for variable {variable}.")

            def model():
                for node in self.nodes:
                    if node in data.columns:
                        cpd = self.get_cpds(node)
                        parents = cpd.parents

                        parent_data = (
                            {p: torch.tensor(data[p].values).float() for p in parents}
                            if parents
                            else None
                        )

                        with pyro.plate(f"plate_{node}", len(data[node])):
                            pyro.condition(
                                cpd.fn(parent_data),
                                data={"data_dist": torch.tensor(data[node].values)},
                            )

            def guide():
                for node in self.nodes:
                    if node in guides:
                        with pyro.plate(f"plate_{node}", len(data[node])):
                            guides[node]()

            gamma = 0.01
            lrd = gamma ** (1 / num_steps)
            optimizer = pyro.optim.ClippedAdam({"lr": learning_rate, "lrd": lrd})
            svi = pyro.infer.SVI(
                model=model,
                guide=guide,
                optim=optimizer,
                loss=pyro.infer.Trace_ELBO(),
            )

            for step in range(num_steps):
                loss = svi.step()
                if step % 250 == 0:
                    print(f"Loss : {loss}")

            params = pyro.get_param_store()
            return {name: params[name].detach().numpy() for name in params.keys()}
        else:
            nuts_kwargs = nuts_kwargs or {}
            mcmc_kwargs = mcmc_kwargs or {}

            def combined_model():
                for node in sort_nodes:
                    if node in data.columns:
                        cpd = self.get_cpds(node)

                        dist_fn = cpd.fn()
                        obs_data = torch.tensor(data[node].values).float()

                        with pyro.plate(f"plate_{node}", len(data)):
                            pyro.sample(node, dist_fn, obs=obs_data)

            nuts_kernel = pyro.infer.NUTS(combined_model, **nuts_kwargs)

            mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_steps, **mcmc_kwargs)
            mcmc.run()

            samples = mcmc.get_samples()

            return samples
