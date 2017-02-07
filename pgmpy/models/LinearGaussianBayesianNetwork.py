from __future__ import division

import numpy as np
import networkx as nx

from pgmpy.models import BayesianModel
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.continuous import JointGaussianDistribution


class LinearGaussianBayesianNetwork(BayesianModel):
    """
    A Linear Gaussain Bayesian Network is a Bayesian Network, all
    of whose variables are continuous, and where all of the CPDs
    are linear Gaussians.

    An important result is that the linear Gaussian Bayesian Networks
    are an alternative representation for the class of multivariate
    Gaussian distributions.

    """

    def add_cpds(self, *cpds):
        """
        Add linear Gaussian CPD (Conditional Probability Distribution)
        to the Bayesian Model.

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
                print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        """
        for cpd in cpds:
            if not isinstance(cpd, LinearGaussianCPD):
                raise ValueError('Only LinearGaussianCPD can be added.')

            if set(cpd.variables) - set(cpd.variables).intersection(
                    set(self.nodes())):
                raise ValueError('CPD defined on variable not in the model', cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning("Replacing existing CPD for {var}".format(var=cpd.variable))
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
                print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
                print(cpd)

        P(x1) = N(1; 4)

        """
        return super(LinearGaussianBayesianNetwork, self).remove_cpds(*cpds)

    def to_joint_gaussian(self):
        """
        The linear Gaussian Bayesian Networks are an alternative
        representation for the class of multivariate Gaussian distributions.
        This method returns an equivalent joint Gaussian distribution.

        Returns
        -------
        JointGaussianDistribution: An equivalent joint Gaussian
                                   distribution for the network.

        Reference
        ---------
        Section 7.2, Example 7.3,
        Probabilistic Graphical Models, Principles and Techniques

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> jgd = model.to_joint_gaussian()
        >>> jgd.variables
        ['x1', 'x2', 'x3']
        >>> jgd.mean
        array([[ 1. ],
               [-4.5],
               [ 8.5]])
        >>> jgd.covariance
        array([[ 4.,  2., -2.],
               [ 2.,  5., -5.],
               [-2., -5.,  8.]])

        """
        variables = nx.topological_sort(self)
        mean = np.zeros(len(variables))
        covariance = np.zeros((len(variables), len(variables)))

        for node_idx in range(len(variables)):
            cpd = self.get_cpds(variables[node_idx])
            mean[node_idx] = sum([coeff * mean[variables.index(parent)] for
                                  coeff, parent in zip(cpd.beta_vector, cpd.evidence)]) + cpd.beta_0
            covariance[node_idx, node_idx] = sum(
                [coeff * coeff * covariance[variables.index(parent), variables.index(parent)]
                 for coeff, parent in zip(cpd.beta_vector, cpd.evidence)]) + cpd.variance

        for node_i_idx in range(len(variables)):
            for node_j_idx in range(len(variables)):
                if covariance[node_j_idx, node_i_idx] != 0:
                    covariance[node_i_idx, node_j_idx] = covariance[node_j_idx, node_i_idx]
                else:
                    cpd_j = self.get_cpds(variables[node_j_idx])
                    covariance[node_i_idx, node_j_idx] = sum(
                        [coeff * covariance[node_i_idx, variables.index(parent)]
                         for coeff, parent in zip(cpd_j.beta_vector, cpd_j.evidence)])

        return JointGaussianDistribution(variables, mean, covariance)

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
                    raise ValueError("CPD associated with %s doesn't have "
                                     "proper parents associated with it." % node)
        return True

    def get_cardinality(self, node):
        """
        Cardinality is not defined for continuous variables.
        """
        raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(self, data, estimator=None, state_names=[], complete_samples_only=True, **kwargs):
        """
        For now, fit method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError("fit method has not been implemented for LinearGaussianBayesianNetwork.")

    def predict(self, data):
        """
        For now, predict method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError("predict method has not been implemented for LinearGaussianBayesianNetwork.")

    def to_markov_model(self):
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError("to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.")

    def is_imap(self, JPD):
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError("is_imap method has not been implemented for LinearGaussianBayesianNetwork.")
