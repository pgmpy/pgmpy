import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.global_vars import logger
from pgmpy.models import BayesianNetwork


class FunctionalBayesianNetwork(BayesianNetwork):
    def add_cpds(self, *cpds):
        """
        Add Functional CPD (Conditional Probability Distribution)
        to the Bayesian Network.

        Parameters
        ----------
        cpds  :  instances of FunctionalCPD
            List of FunctionalCPDs which will be associated with the model
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
        """
        if seed is not None:
            np.random.seed(seed)

        nodes = list(nx.topological_sort(self))
        samples = pd.DataFrame(index=range(n_samples))

        for node in nodes:
            cpd = self.get_cpds(node)
            parent_samples = samples[cpd.parents] if cpd.parents else None
            samples[node] = cpd.sample(
                n_samples=n_samples, parent_sample=parent_samples
            )

        return samples


FBNetwork = FunctionalBayesianNetwork([("x1", "x3"), ("x2", "x3")])

x1_cpd = FunctionalCPD(variable="x1", fn=lambda _: np.random.normal(0, 1))
x2_cpd = FunctionalCPD("x2", lambda _: np.random.normal(0, 1))
x3_cpd = FunctionalCPD(
    variable="x3",
    fn=lambda parent_sample: np.random.normal(
        1.0 + 0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"], 1.0
    ),
    parents=["x1", "x2"],
)

FBNetwork.add_cpds(x1_cpd, x2_cpd, x3_cpd)
# print(FBNetwork.check_model())
# print(FBNetwork.simulate())
