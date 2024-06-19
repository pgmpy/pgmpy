import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.distributions import GaussianDistribution
from pgmpy.global_vars import logger
from pgmpy.models import BayesianNetwork


class LinearGaussianBayesianNetwork(BayesianNetwork):
    """
    A Linear Gaussian Bayesian Network is a Bayesian Network, all
    of whose variables are continuous, and where all of the CPDs
    are linear Gaussians.

    An important result is that the linear Gaussian Bayesian Networks
    are an alternative representation for the class of multivariate
    Gaussian distributions.

    """

    def add_cpds(self, *cpds):
        """
        Add linear Gaussian CPD (Conditional Probability Distribution)
        to the Bayesian Network.

        Parameters
        ----------
        cpds  :  instances of LinearGaussianCPD
            List of LinearGaussianCPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.cpds:
        ...     print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        """
        for cpd in cpds:
            if not isinstance(cpd, LinearGaussianCPD):
                raise ValueError("Only LinearGaussianCPD can be added.")

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
        A list of linear Gaussian CPDs.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        """
        return super(LinearGaussianBayesianNetwork, self).get_cpds(node)

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: LinearGaussianCPD object
            A LinearGaussianCPD object on any subset of the variables
            of the model which is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)

        P(x1) = N(1; 4)

        """
        return super(LinearGaussianBayesianNetwork, self).remove_cpds(*cpds)

    def get_random_cpds(self, loc=0, scale=1, seed=None):
        """
        Generates random Linear Gaussian CPDs for the model. The coefficients
        are sampled from a normal distribution with mean `loc` and standard
        deviation `scale`.

        Parameters
        ----------
        loc: float
            The mean of the normal distribution from which the coefficients are
            sampled.

        scale: float
            The standard deviation of the normal distribution from which the
            coefficients are sampled.

        seed: int
            The seed for the random number generator.
        """
        rng = np.random.default_rng(seed=seed)

        cpds = []
        for var in self.nodes():
            parents = self.get_parents(var)
            cpds.append(
                LinearGaussianCPD(
                    var,
                    evidence_mean=rng.normal(
                        loc=loc, scale=scale, size=(len(parents) + 1)
                    ),
                    evidence_variance=abs(rng.normal(loc=loc, scale=scale)),
                    evidence=parents,
                )
            )
        return cpds

    def to_joint_gaussian(self):
        """
        Linear Gaussian Bayesian Networks can be represented using a joint
        Gaussian distribution over all the variables. This method gives
        the mean and covariance of this equivalent joint gaussian distribution.

        Returns
        -------
        mean, cov: np.ndarray, np.ndarray
            The mean and the covariance matrix of the joint gaussian distribution.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> mean, cov = model.to_joint_gaussian()
        >>> mean
        array([ 1. ], [-4.5], [ 8.5])
        >>> cov
        array([[ 4.,  2., -2.],
               [ 2.,  5., -5.],
               [-2., -5.,  8.]])

        """
        variables = list(nx.topological_sort(self))
        var_to_index = {var: i for i, var in enumerate(variables)}
        n_nodes = len(self.nodes())

        # Step 1: Compute the mean for each variable.
        mean = {}
        for var in variables:
            cpd = self.get_cpds(node=var)
            mean[var] = (
                cpd.mean * (np.array([1] + [mean[u] for u in cpd.evidence]))
            ).sum()
        mean = np.array([mean[u] for u in variables])

        # Step 2: Populate the adjacency matrix, and variance matrix
        B = np.zeros((n_nodes, n_nodes))
        omega = np.zeros((n_nodes, n_nodes))
        for var in variables:
            cpd = self.get_cpds(node=var)
            for i, evidence_var in enumerate(cpd.evidence):
                B[var_to_index[evidence_var], var_to_index[var]] = cpd.mean[i + 1]
            omega[var_to_index[var], var_to_index[var]] = cpd.variance

        # Step 3: Compute the implied covariance matrix
        I = np.eye(n_nodes)
        inv = np.linalg.inv((I - B))
        implied_cov = inv.T @ omega @ inv

        # Round because numerical errors can lead to non-symmetric cov matrix.
        return mean.round(decimals=8), implied_cov.round(decimals=8)

    def simulate(self, n=1000, seed=None):
        """
        Simulates data from the given model.

        Parameters
        ----------
        n: int
            The number of samples to draw from the model.

        seed: int (default: None)
            Seed for the random number generator.

        Returns
        -------
        pandas.DataFrame: generated samples
            A pandas data frame with the generated samples.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.simulate(n=500, seed=42)
        """
        if len(self.cpds) != len(self.nodes()):
            raise ValueError(
                "Each node in the model should have a CPD associated with it"
            )

        mean, cov = self.to_joint_gaussian()
        variables = list(nx.topological_sort(self))
        rng = np.random.default_rng(seed=seed)
        return pd.DataFrame(
            rng.multivariate_normal(mean=mean, cov=cov, size=n), columns=variables
        )

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

            if isinstance(cpd, LinearGaussianCPD):
                if set(cpd.evidence) != set(self.get_parents(node)):
                    raise ValueError(
                        "CPD associated with %s doesn't have "
                        "proper parents associated with it." % node
                    )
        return True

    def get_cardinality(self, node):
        """
        Cardinality is not defined for continuous variables.
        """
        raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(self, data, method="mle"):
        """
        Estimates the parameters of the model using the given `data`.
        """
        cpds = []
        for node in self.nodes():
            parents = self.get_parents(node)
            if len(parents) == 0:
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        evidence_mean=[data.loc[:, node].mean()],
                        evidence_variance=data.loc[:, node].var(),
                    )
                )
            else:
                lm = LinearRegression().fit(data.loc[:, parents], data.loc[:, node])
                error_var = (data.loc[:, node] - lm.predict(data.loc[:, parents])).var()
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        evidence_mean=np.append([lm.intercept_], lm.coef_),
                        evidence_variance=error_var,
                        evidence=parents,
                    )
                )

        self.add_cpds(*cpds)

    def predict(self, data):
        """
        For now, predict method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "predict method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def to_markov_model(self):
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "to_markov_model method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def is_imap(self, JPD):
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "is_imap method has not been implemented for LinearGaussianBayesianNetwork."
        )
