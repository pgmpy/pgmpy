# -*- coding: utf-8 -*-

from __future__ import division
import warnings

import numpy as np
from numpy import linalg
from scipy.stats import multivariate_normal

from pgmpy.factors.distributions import BaseDistribution, BaseConditionalDistribution


# TODO: Add support for non linear Gaussian Distributions
class GaussianDistribution(BaseDistribution):
    """
    Class representing Gaussian Distributions (Joint, Linear Conditional and
    Non-linear Conditional).
    """
    def __init__(self, variables, mean, cov, evidence=None):
        """
        Base class for representing Gaussian Distributions.

        Joint Gaussian Distributions are characterized by two parameters: the
        mean vector and the covariance matrix. For example:

            P(X, Y) = N([mean_x, mean_y]; [[var_x, covar_xy], [covar_yx, var_y]])

        Linear Conditional Gaussian Distribution assumes that the mean of variables
        depends linearly on the evidence and the variance is independent of the
        evidence i.e constant. For example:

            P(X, Y | A) = N([-2A + 0.9, -3A + 0.3]; [[1, 0.3], [0.3, 1]])

        Here, the mean of the variables X and Y is linearly dependent on the
        evidence A and the covariance matrix is constant.

        Non-Linear Conditional Gaussian Distribution doesn't make any assumptions
        about the relation between the variables' means and their evidence. But
        the covariance matrix is still considered to be constant and not
        dependent on the evidence. For example:

            P(X, Y | A) = N([-2A^2 + 0.9, -3A^3 + 2A + 0.3]; [[1, 0.3], [0.3, 1]])

        Parameters
        ----------
        variables: str or list, array-like (size: n)
                A list of hashable python objects representing the names of the
                variables on which the conditional distribution is defined.

        mean: list, array-like (size: n x (m+1))
                An array of int or float representing the coefficient vector of
                the means of each variable in variables.

        cov: list, array-like (size: n x n)
                A 2-D array representing the covariance matrix of the conditional
                distribution.

        evidence: str or list, array-like (size: m)
                A list of hashable python objects representing the names of the
                evidences in the conditional distribution.

        Examples
        --------
        For representing a simple Joint Gaussian Distribution:

            P(X, Y) = N([1, 2]; [[0.3, 0.4], [0.4, 0.8]])

        >>> dist_joint = GaussianDistribution(
        ...                     variables=['X', 'Y'],
        ...                     mean=[1, 2],
        ...                     cov=[[0.3, 0.4], [0.4, 0.8]])

        For representing a Conditional Gaussian Distribution:

            P(X | Y, Z) = N(2Y + 3Z + 0.6; 1)

        >>> dist_single = GaussianDistribution(
        ...                     variables='X',
        ...                     mean=[2, 3, 0.6],
        ...                     cov=1,
        ...                     evidence=['Y', 'Z'])
        >>> dist_single.variables
        ['X']
        >>> dist_single.mean
        [[2, 3, 0.6]]
        >>> dist_single.cov
        [[1]]

        For representing a Conditional Gaussian Distribution with multiple
        variables:

            P(X, Y | A, B) = N([0.3A + 0.5B + 0.8, 0.4A + 0.2B + 0.7], [[1, 0], [0, 1]]
        >>> dist_multi = GaussianDistribution(
        ...                     variables=['X', 'Y'],
        ...                     mean=[[0.3, 0.5, 0.8], [0.4, 0.2, 0.7]],
        ...                     cov=[[1, 0], [0, 1]],
        ...                     evidence=['A', 'B'])
        >>> dist_multi.evidence
        ['A', 'B']
        >>> dist_multi.mean
        [[0.3, 0.5, 0.8],
         [0.4, 0.2, 0.7]]

        Returns
        -------
        pgmpy.factors.GaussianDistribution object.

        See Also
        --------
        GaussianDistribution
        ContinuousFactor
        """
        self.variables = np.atleast_1d(variables)
        self.cov = np.atleast_2d(cov)
        # TODO: Check if cov is a valid covariance matrix.

        # Checking for len(evidence) in case an empty array is passed.
        if evidence is not None and len(evidence):
            self.evidence = np.atleast_1d(evidence)
        else:
            self.evidence = np.array([])

        # Reshaping because mean can be either 1-D or 2-D
        self.mean = np.atleast_2d(mean).reshape(self.variables.size, self.evidence.size + 1)

        # Both variables and evidence should be either a str or 1-D array.
        if not self.variables.ndim == 1:
            raise ValueError("variables: Expected str or 1-D list/array, got: {size}".format(
                size=self.variables.ndim))
        elif not self.evidence.ndim == 1:
            raise ValueError("evidence: Expected str or 1-D list/array, got: {size}".format(
                size=self.evidence.ndim))

        # Check whether the size of mean and cov matches with the variables and evidence
        elif not self.mean.shape == (self.variables.size, self.evidence.size + 1):
            raise ValueError("mean: Expected array of shape: {ex_shape}, got: {tr_shape}".format(
                ex_shape=(self.variables.size, self.evidence.size+1), tr_shape=self.mean.shape))
        elif not self.cov.shape == (self.variables.size, self.variables.size):
            raise ValueError(
                "cov: Expected scalar/array of shape: {ex_shape}, got: {tr_shape}".format(
                    ex_shape=(self.variables.size, self.variables.size),
                    tr_shape=self.cov.shape))

    def pdf(self, evidence_values=None):
        """
        Returns the probability density function(pdf).

        Parameters
        ----------
        evidence_values: array_like, 1-D (optional)
                The values of the evidence variables for which the pdf
                is to be computed.

        Returns
        -------
        function: The probability density function of the distribution.

        Examples
        --------
        >>> from pgmpy.factors.distribution import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['X', 'Y'],
        ...                             mean=[[1, 1, 0.5],
        ...                                   [2, 1, 0.3]],
        ...                             cov=[[1, 0], [0, 1]],
        ...                             evidence=['A', 'B'])
        >>> dist.pdf
        >>> dist.pdf(var_values=[0, 0], evidence_values=[1, 0.4])
        """
        if self.evidence.size:
            if not evidence_values:
                raise ValueError("evidence_values is required for pdf ",
                                 "of a conditional distribution")
            else:
                evidence_values = np.atleast_1d(evidence_values)
                if evidence_values.size != self.evidence.size:
                    raise ValueError("evidence_values: Expected size: {exp}, ",
                                     "got: {real}".format(
                                         exp=self.evidence.shape,
                                         real=evidence_values.shape))
                else:
                    # Appending a 1 to evidence_values to take care of constant
                    # term in the mean.
                    mean = np.sum(self.mean * np.append(evidence_values, 1),
                                  axis=1, keepdims=True)
        else:
            mean = self.mean

        def _pdf(x):
            """
            Returns the pdf value of the distribution at point x.

            Parameters
            ----------
            x : array-like
               The values of the variables at which the pdf is to be computed.

            Returns
            -------
            float/np.ndarray : The pdf value/values of the distribution at
                the points in x.

            Examples
            --------
            >>> _pdf([1, 2, 3]) # dist with 3 variables
            0.063493635934240983
            >>> _pdf([[1, 2, 3], [1, 2, 3]])
            array([ 0.06349364,  0.06349364])
            """
            return multivariate_normal.pdf(x, mean=mean.flatten(), cov=self.cov)

        return _pdf

    def assignment(self, x, evidence_values=None):
        """
        Returns the probability value of the distribution at the given point.

        Parameters
        ----------
        x : array-like
                The point(s) at which the probability value(s) of the distribution
                needs to be computed. The length of x should be equal to the
                number of variables in the distribution.

        evidence_values: array_like, 1-D (optional)
                The values of the evidence variables for which the pdf
                is to be computed.

        Returns
        -------
        float/array-like : Returns a float value if a x is a 1-D array.
                           Returns an array of floats if x is a 2-D array.

        Examples
        --------
        >>> from pgmpy.factors.distributions import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['X', 'Y', 'Z'],
                                        mean=[1, 1, 1],
                                        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> points = [[-0.34098948,  1.56544901,  0.44807747],
                      [ 0.3504427 , -0.43818711, -0.73991183],
                      [-0.29182544,  0.9664683 , -0.13819649]]
        >>> dist.assignment(points)
        array([0.018909, 0.00402345, 0.01441437])
        """
        density_function = self.pdf(evidence_values=evidence_values)
        return density_function(x)

    def copy(self):
        """
        Returns a copy of the distribution.

        Returns
        -------
        GaussianDistribution instance: Copy of the distribution

        Examples
        --------
        >>> from pgmpy.factors.distributions import GaussianDistribution
        >>> dist = GaussianDistribution(
        ...                      variables='X',
        ...                      mean=[0.2, 0.4, 0.2],
        ...                      cov=1,
        ...                      evidence=['Y', 'Z'])
        >>> dist_copy = dist.copy()
        >>> dist_copy.variables
        ['X']
        >>> dist_copy.evidence
        ['Y', 'Z']
        >>> dist_copy.mean
        [[0.2, 0.4, 0.2]]
        """
        return GaussianDistribution(variables=self.variables.copy(),
                                    mean=self.mean.copy(),
                                    cov=self.cov.copy(),
                                    evidence=self.evidence.copy())

    def __str__(self):
        """
        Returns a meaningful str when print is called on the object.
        """
        if self.evidence.size:
            evi = np.append(self.evidence, '')
            means = [' + '.join([str(i)+j for i, j in zip(self.mean[index], evi)])
                     for index in range(self.mean.shape[0])]
            string = "P({variables} | {evidence}) = N({mean}; {cov})".format(
                variables=', '.join(self.variables),
                evidence=', '.join(self.evidence),
                mean=', '.join(means),
                cov=', '.join([str(i) for i in self.cov]))
        else:
            string = "P({variables}) = N({mean}; {cov})".format(
                variables=', '.join(self.variables),
                mean=', '.join([str(i) for i in self.mean.flatten()]),
                cov=', '.join([str(i) for i in self.cov]))

        return string

    def __repr__(self):
        """
        Meaningful short str when __repr__ is called.
        """
        if self.evidence.size:
            string = "P({variables} | {evidence})".format(
                variables=', '.join(self.variables),
                evidence=', '.join(self.evidence))
        else:
            string = "P({variables})".format(
                variables=', '.join(self.variables))
        return "<GaussianDistribution representing {string} at {address}>".format(
            string=string, address=hex(id(self)))

    def _reorder_vars(self, variables=None, evidence=None, inplace=True):
        """
        Modifies the mean and the cov arrays for new order of
        variables and evidence variables.

        Parameters
        ----------
        variables: array-like (1-D), optional
                The new order of variables in the distribution.

        evidence: array_like (1-D), optional
                The new order of evidence variables in the distribution.

        inplace: boolean (default: True)
                If True, modifies the distribution itself.
                If False, returns a copy of the distribution with the new
                        variable orders.

        Returns
        -------
        None or GaussianDistribution:
            If inplace=True, returns None
            If inplace=False, returns GaussianDistribution

        Examples
        --------
        >>> from pgmpy.factors import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['X', 'Y'],
        ...                                 mean=[1, 2],
        ...                                 cov=[[0.8, 0.5], [0.5, 0.3]])
        >>> reverse_var = dist._reorder_vars(variables=['Y', 'X'])
        >>> reverse_var.mean
        np.array([[2], [1]])
        >>> reverse_var.cov
        np.array([[0.3, 0.5], [0.5, 0.8]])
        """
        # Check if the arguments have all the variables
        if (variables is not None) and (sorted(variables) != sorted(self.variables)):
                raise ValueError(
                    "variables should have all the variables of the distribution")
        if (evidence is not None) and (sorted(evidence) != sorted(self.evidence)):
            raise ValueError(
                "evidence should have all the evidence variables of the distribution")

        dist = self if inplace else self.copy()

        # Modify the values for the new orders.
        if variables is not None:
            # Compute an index array on old order for the new order.
            new_var_index = []
            variables_list = list(self.variables)
            for var in variables:
                new_var_index.append(variables_list.index(var))

            dist.mean = dist.mean[new_var_index, :]
            dist.cov = dist.cov[new_var_index, :][:, new_var_index]
            dist.variables = np.array(variables)

        if evidence is not None:
            # Compute an index array on old order for the new order.
            new_evidence_index = []
            evidence_list = list(self.evidence)
            for evi in evidence:
                new_evidence_index.append(evidence_list.index(evi))
            new_evidence_index.append(-1)  # To take care of the constant in mean

            dist.mean = dist.mean[:, new_evidence_index]
            dist.evidence = np.array(evidence)

        if not inplace:
            return dist

    def marginalize(self, variables, inplace=True):
        """
        Marginalize the distribution over one or more variables.

        Parameters
        ----------
        variables: array-like, iterable
                List of variables over which the marginalization is to be done.

        inplace: boolean
                if inplace=True (default) returns None
                if inplace=False returns a new GaussianDistribution instance

        Returns
        -------
        None or GaussianDistribution instance:
                if inplace=True (default) returns None
                if inplace=False returns GaussianDistribution instance.

        Examples
        --------
        >>> from pgmpy.factors import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['X', 'Y', 'Z'],
        ...                             mean=[[0.4, 0.3, 0.8],
        ...                                   [0.8, 0.7, 0.1],
        ...                                   [0.9, 0.4, 0.7]],
        ...                             cov=[[0.8, 0.9, 0.3],
        ...                                  [0.7, 0.5, 0.6],
        ...                                  [0.4, 0.2, 0.4]],
        ...                             evidence=['A', 'B'])
        >>> dist.marginalize(['Y'])
        >>> dist.variables
        ['X', 'Z']
        >>> dist.mean
        [[0.4, 0.3, 0.8], [0.9, 0.4, 0.7]]
        >>> dist.cov
        [[0.8, 0.3], [0.4, 0.4]]
        """
        dist = self if inplace else self.copy()

        variables = list(variables)

        # Computing variable when we need to keep after marginalizing
        index_to_keep = []
        vars_to_keep = []
        for index, var in enumerate(self.variables):
            if var not in variables:
                vars_to_keep.append(var)
                index_to_keep.append(index)

        # Throw a warning if not all the variables are in the distribution
        if len(self.variables) - len(index_to_keep) != len(variables):
            warnings.warn("Possible repetition in variables or not all " +
                          "variables are in the distribution")

        # Compute the new mean and covariance
        dist.variables = np.array(vars_to_keep)
        dist.mean = dist.mean[index_to_keep, :]
        dist.cov = dist.cov[np.ix_(index_to_keep, index_to_keep)]

        if not inplace:
            return dist

    def normalize(self, inplace=True):
        """
        Normalizes the distribution. But in case of a Gaussian Distribution
        the distribution is always normalized. This method is here just to
        have uniform API over all the BaseFactor subclasses.

        Parameters
        ----------
        inplace: boolean (default: True)
                If True, returns None.
                If False, returns a copy of the object.
        """
        dist = self if inplace else self.copy()

        if not inplace:
            return dist

    def reduce(self, variables, inplace=True):
        """
        Reduce the variables from the distribution.

        Parameters
        ----------
        variables: array-like, (size: n x 2)
                List of variables which will be reduced to specific values in
                the distribution. The array is assumed to be 2-D of the form
                [(var1, value1), (var2, value2), ...].

        inplace: boolean (default:True)
                If True, reduces the variables, and makes changes to the instance
                        called on.
                If False, creates a copy of the instance and reduces variables and
                returns it.

        Returns
        -------
        None or GaussianDistribution:
                If inplace=True, returns None.
                If inplace=False, returns GaussianDistribution

        Examples
        --------
        >>> from pgmpy.factors import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['X', 'Y', 'Z'],
        ...                             mean=[[0.4, 0.3, 0.8],
        ...                                   [0.8, 0.7, 0.1],
        ...                                   [0.9, 0.4, 0.7]],
        ...                             cov=[[0.8, 0.9, 0.3],
        ...                                  [0.7, 0.5, 0.6],
        ...                                  [0.4, 0.2, 0.4]],
        ...                             evidence=['A', 'B'])
        >>> dist.reduce(['Y', 0.5])
        >>> dist.variables
        ['X', 'Z']
        >>> dist.mean
        [[0.4, 0.3, 0.8], [0.9, 0.4, 0.7]]
        >>> dist.cov
        [[0.8, 0.3], [0.4, 0.4]]
        """
        dist = self if inplace else self.copy()

        variables = np.array(variables)
        variables_shape = variables.ndim
        if variables_shape not in [1, 2]:
            raise ValueError("variables should either be 1-D or 2-D")

        # Case when there is no value given for conditional variables.
        # \mu_{a|b} = \mu_a - \Lambda_{aa}^{-1] \Lambda_{ab} y_b
        # where y_b[:, 0] = [1]
        #       y_b[:, 1:] = \mu_b

        if variables_shape == 1:
            index_to_keep = []
            index_to_remove = []
            for index, var in enumerate(self.variables):
                if var in variables:
                    index_to_remove.append(index)
                else:
                    index_to_keep.append(index)

            no_keep = len(index_to_keep)
            no_remove = len(index_to_remove)

            # Compute new values
            precision = linalg.inv(self.cov)
            precision_aa = precision[np.ix_(index_to_keep, index_to_keep)]
            precision_ab = precision[np.ix_(index_to_keep, index_to_remove)]
            mean_a = self.mean[index_to_keep, :]
            mean_b = self.mean[index_to_remove, :]
            new_mean = np.zeros((no_keep, len(dist.evidence) + no_remove + 1))

            # Computing \Lambda_{aa}^{-1} * \Lambda_{ab}
            temp = np.dot(linalg.inv(precision_aa), precision_ab)
            new_mean[:, no_remove:] = mean_a + np.dot(temp, mean_b)
            new_mean[:, :no_remove] = -temp
            dist.mean = new_mean
            dist.cov = linalg.inv(precision_aa)

            dist.variables = self.variables[index_to_keep]
            dist.evidence = np.append(self.variables[index_to_remove], dist.evidence)

        # Case when the value of the conditional variables are given.
        # \mu_{a|b} = \mu_a - \Lamba_{aa}^{-1} \Lambda_{ab} y_b
        # where y_b[:, -1] = x_b - \mu_b[:, -1]
        #       y_b[:, :-1] = \mu_b[:, :-1]
        #       \Lambda = \Sigma^{-1}

        if variables_shape == 2:
            vars_to_remove, values = list(zip(*variables))
            values = np.array(values, dtype=float)

            # Compute indexes to keep and indexes to remove
            index_to_keep = []
            index_to_remove = []
            for index, var in enumerate(self.variables):
                if var in vars_to_remove:
                    index_to_remove.append(index)
                else:
                    index_to_keep.append(index)

            # Compute new values
            precision = linalg.inv(self.cov)
            precision_aa = precision[np.ix_(index_to_keep, index_to_keep)]
            precision_ab = precision[np.ix_(index_to_keep, index_to_remove)]
            mean_a = self.mean[index_to_keep, :]
            mean_b = self.mean[index_to_remove, :]

            # Using mean_b[:, -1] to handle to case of conditional distributions when
            # the value will be substracted from the constant term of mean.
            x_b_mean_b = mean_b.copy()
            x_b_mean_b[:, -1] = values - mean_b[:, -1]

            dist.cov = linalg.inv(precision_aa)
            # Using mean_b[:, -1] to handle to case of conditional distributions when
            # the value will be substracted from the constant term of mean.
            dist.mean = mean_a - np.dot(np.dot(linalg.inv(precision_aa),
                                               precision_ab), x_b_mean_b)
            dist.variables = self.variables[index_to_keep]

        if not inplace:
            return dist

    def _add_independent_variables(self, variables=None, evidence=None, inplace=False):
        """
        Add new independent variables to the distribution.

        Parameters
        ----------
        variables: list, iterable (shape: n x 3)
                The variables to add to the distribution. It should be of the form:
                [(var1, mean1, cov1), (var2, mean2, cov2), ...]

        evidence: list, iterable
                The variables to add as conditional variables to the distribution.

        inplace: boolean
                If inplace=True, modifies the distribution else returns a new
                distribution.

        Returns
        -------
        None or GaussianDistribution instance:
                If inplace=True, returns GaussianDistribution
                If inplace=False, returns None

        Examples
        --------
        >>> from pgmpy.factors import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['X', 'Y', 'Z'],
        ...                             mean=[[0.4, 0.3, 0.8],
        ...                                   [0.8, 0.7, 0.1],
        ...                                   [0.9, 0.4, 0.7]],
        ...                             cov=[[0.8, 0.9, 0.3],
        ...                                  [0.7, 0.5, 0.6],
        ...                                  [0.4, 0.2, 0.4]],
        ...                             evidence=['A', 'B'])
        >>> dist._add_independent_variables(variables=[('L', 1, 1), ('M', 2, 2)],
        ...                                 evidence=['C', 'D'], inplace=True)
        >>> dist.variables
        array(['X', 'Y', 'Z', 'L', 'M'], dtype='<U1')
        >>> dist.evidence
        array(['C', 'D', 'A', 'B'], dtype='<U1')
        >>> dist.mean
        array([[ 0. ,  0. ,  0.4,  0.3,  0.8],
               [ 0. ,  0. ,  0.8,  0.7,  0.1],
               [ 0. ,  0. ,  0.9,  0.4,  0.7],
               [ 0. ,  0. ,  0. ,  0. ,  1. ],
               [ 0. ,  0. ,  0. ,  0. ,  2. ]])
        >>> dist.cov
        array([[ 0.8,  0.9,  0.3,  0. ,  0. ],
               [ 0.7,  0.5,  0.6,  0. ,  0. ],
               [ 0.4,  0.2,  0.4,  0. ,  0. ],
               [ 0. ,  0. ,  0. ,  1. ,  0. ],
               [ 0. ,  0. ,  0. ,  0. ,  2. ]])
        """
        dist = self if inplace else self.copy()

        if evidence:
            new_mean = np.zeros((len(dist.variables), len(dist.evidence)+len(evidence)+1),
                                dtype=np.float)
            new_mean[:, len(evidence):] = dist.mean
            dist.mean = new_mean
            dist.evidence = np.append(evidence, dist.evidence)

        if variables:
            for var in variables:
                mean = np.zeros((1, len(dist.evidence)+1))
                mean[0, -1] = var[1]
                dist.mean = np.append(dist.mean, mean, axis=0)
                cov = np.zeros((len(dist.variables)+1, len(dist.variables)+1))
                cov[:-1, :-1] = dist.cov
                cov[-1, -1] = var[2]
                dist.cov = cov
                dist.variables = np.append(dist.variables, var[0])

        if not inplace:
            return dist

    def product(self, dist, inplace=False):
        pass

    def divide(self, dist, inplace=False):
        pass

    def __div__(self, other):
        pass

    def __mult__(self, other):
        pass
