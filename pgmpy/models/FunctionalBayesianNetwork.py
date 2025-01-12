import networkx as nx
import numpy as np
import pandas as pd
import pyro

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
        >>> import numpy as np

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: np.random.normal(0, 1))
        >>> cpd2 = FunctionalCPD("x2", lambda parent: np.random.normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: np.random.normal(parent["x2"] + 0.3, 2), parents=["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)

        """
        for cpd in cpds:
            if not isinstance(cpd, FunctionalCPD):
                raise ValueError(
                    f"Only FunctionalCPD instances can be added. Got {type(cpd)}"
                )

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError(
                    f"CPD defined on variable that is not present in the model: {cpd}"
                )

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

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: np.random.normal(0, 1))
        >>> cpd2 = FunctionalCPD("x2", lambda parent: np.random.normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: np.random.normal(parent["x2"] + 0.3, 2), parents=["x2"])
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

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: np.random.normal(0, 1))
        >>> cpd2 = FunctionalCPD("x2", lambda parent: np.random.normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: np.random.normal(parent["x2"] + 0.3, 2), parents=["x2"])
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

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: np.random.normal(0, 1))
        >>> cpd2 = FunctionalCPD("x2", lambda parent: np.random.normal(parent["x1"] + 2.0, 1), parents=["x1"])
        >>> cpd3 = FunctionalCPD("x3", lambda parent: np.random.normal(parent["x2"] + 0.3, 2), parents=["x2"])
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
