# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal

from pgmpy.factors.distributions import BaseDistribution


class GaussianDistribution(BaseDistribution):
    """
    In its most common representation, a multivariate Gaussian distribution
    over X1, X2, ..., Xn is characterized by an n-dimensional mean vector μ,
    and a symmetric n x n covariance matrix Σ.

    This is the base class for its representation.
    """
    def __init__(self, variables, mean, cov):
        """
        Parameters
        ----------
        variables: iterable of any hashable python object
            The variables for which the distribution is defined.

        mean: list, array-like
            1-D array of size n where n is the number of variables.

        cov: n x n, 2-D array like
            n x n dimensional matrix where n is the number of variables.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis = GD(variables=['x1', 'x2', 'x3'],
        ...          mean=np.array([1, -3, 4]),
        ...          cov=np.array([[4, 2, -2],
        ...                        [2, 5, -5],
        ...                        [-2, -5, 8]]))
        >>> dis.variables
        ['x1', 'x2', 'x3']
        >>> dis.mean
        array([[ 1],
               [-3],
               [4]]))
        >>> dis.cov
        array([[4, 2, -2],
               [2, 5, -5],
               [-2, -5, 8]])
        >>> dis.assignment([0, 0, 0])
        0.0014805631279234139
        """
        no_of_var = len(variables)

        self.variables = variables
        self.mean = np.asarray(np.reshape(mean, (no_of_var, 1)), dtype=float)
        self.covariance = np.asarray(cov, dtype=float)
        self._precision_matrix = None

        if len(mean) != no_of_var:
            raise ValueError("Length of mean_vector must be equal to the",
                             "number of variables.")

        if self.covariance.shape != (no_of_var, no_of_var):
            raise ValueError("The Covariance matrix should be a square matrix",
                             " with order equal to the number of variables. ",
                             "Got: {got_shape}, Expected: {exp_shape}".format
                             (got_shape=self.covariance.shape,
                              exp_shape=(no_of_var, no_of_var)))

    @property
    def pdf(self):
        """
        Returns the probability density function(pdf).

        Returns
        -------
        function: The probability density function of the distribution.

        Examples
        --------
        >>> from pgmpy.factors.distributions import GaussianDistribution
        >>> dist = GD(variables=['x1', 'x2', 'x3'],
        ...           mean=[1, -3, 4],
        ...           cov=[[4, 2, -2],
        ...                [2, 5, -5],
        ...                [-2, -5, 8]])
        >>> dist.pdf
        <function pgmpy.factors.distributions.GaussianDistribution.GaussianDistribution.pdf.<locals>.<lambda>>
        >>> dist.pdf([0, 0, 0])
        0.0014805631279234139
        """
        return lambda *args: multivariate_normal.pdf(
            args, self.mean.reshape(1, len(self.variables))[0], self.covariance)

    def assignment(self, *x):
        """
        Returns the probability value of the PDF at the given parameter values.

        Parameters
        ----------
        *x: int, float
            The point at which the value of the pdf needs to be computed. The
            number of values passed should be equal to the number of variables
            in the distribution.

        Returns
        -------
        float: float
            The probability value at the point.

        Examples
        --------
        >>> from pgmpy.factors.distributions import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['x1', 'x2'],
        ...                             mean=[0, 0],
        ...                             cov=[[1, 0],
                                             [0, 1]])
        >>> dist.assignment(0, 0)
        0.15915494309189535
        """
        return self.pdf(*x)

    @property
    def precision_matrix(self):
        """
        Returns the precision matrix of the distribution.

        Precision is defined as the inverse of the variance. This method returns
        the inverse matrix of the covariance.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis = GD(variables=['x1', 'x2', 'x3'],
        ...          mean=[1, -3, 4],
        ...          cov=[[4, 2, -2],
        ...               [2, 5, -5],
        ...               [-2, -5, 8]]))
        >>> dis.precision_matrix
        array([[ 0.3125    , -0.125     ,  0.        ],
               [-0.125     ,  0.58333333,  0.33333333],
               [ 0.        ,  0.33333333,  0.33333333]])
        """
        if self._precision_matrix is None:
            self._precision_matrix = np.linalg.inv(self.covariance)
        return self._precision_matrix

    def marginalize(self, variables, inplace=True):
        """
        Modifies the distribution with marginalized values.

        Parameters
        ----------
        variables: iterator over any hashable object.
                List of variables over which marginalization is to be done.

        inplace: boolean
                If inplace=True it will modify the distribution itself,
                else would return a new distribution.

        Returns
        -------
        GaussianDistribution or None :
                if inplace=True (default) returns None
                if inplace=False return a new GaussianDistribution instance

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis = GD(variables=['x1', 'x2', 'x3'],
        ...          mean=[1, -3, 4],
        ...          cov=[[4, 2, -2],
        ...               [2, 5, -5],
        ...               [-2, -5, 8]]))
        >>> dis.variables
        ['x1', 'x2', 'x3']
        >>> dis.mean
        array([[ 1],
               [-3],
               [ 4]])
        >>> dis.covariance
        array([[ 4,  2, -2],
               [ 2,  5, -5],
               [-2, -5,  8]])

        >>> dis.marginalize(['x3'])
        dis.variables
        ['x1', 'x2']
        >>> dis.mean
        array([[ 1.],
               [-3.]]))
        >>> dis.covariance
        array([[4., 2.],
               [2., 5.]])
        """
        if not isinstance(variables, list):
            raise TypeError("variables: Expected type list or array-like,"
                            "got type {var_type}".format(
                                var_type=type(variables)))

        phi = self if inplace else self.copy()

        index_to_keep = [self.variables.index(var) for var in self.variables
                         if var not in variables]

        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.mean = phi.mean[index_to_keep]
        phi.covariance = phi.covariance[np.ix_(index_to_keep, index_to_keep)]
        phi._precision_matrix = None

        if not inplace:
            return phi

    def reduce(self, values, inplace=True):
        """
        Reduces the distribution to the context of the given variable values.

        The formula for the obtained conditional distribution is given by -

        For,
        .. math:: N(X_j | X_i = x_i) ~ N(mu_{j.i} ; sig_{j.i})

        where,
        .. math:: mu_{j.i} = mu_j + sig_{j, i} * {sig_{i, i}^{-1}} * (x_i - mu_i)
        .. math:: sig_{j.i} = sig_{j, j} - sig_{j, i} * {sig_{i, i}^{-1}} * sig_{i, j}

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_value).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new ContinuosFactor object.

        Returns
        -------
        GaussianDistribution or None:
                if inplace=True (default) returns None
                if inplace=False returns a new GaussianDistribution instance.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis = GD(variables=['x1', 'x2', 'x3'],
        ...             mean=[1, -3, 4],
        ...             cov=[[4, 2, -2],
        ...                  [2, 5, -5],
        ...                  [-2, -5, 8]])
        >>> dis.variables
        ['x1', 'x2', 'x3']
        >>> dis.mean
        array([[ 1.],
               [-3.],
               [ 4.]])
        >>> dis.covariance
        array([[ 4.,  2., -2.],
               [ 2.,  5., -5.],
               [-2., -5.,  8.]])

        >>> dis.reduce([('x1', 7)])
        >>> dis.variables
        ['x2', 'x3']
        >>> dis.mean
        array([[ 0.],
               [ 1.]])
        >>> dis.covariance
        array([[ 4., -4.],
               [-4.,  7.]])

        """
        if not isinstance(values, list):
            raise TypeError("values: Expected type list or array-like, ",
                            "got type {var_type}".format(
                                var_type=type(values)))

        phi = self if inplace else self.copy()

        var_to_reduce = [var for var, value in values]

        # index_to_keep -> j vector
        index_to_keep = [self.variables.index(var) for var in self.variables
                         if var not in var_to_reduce]
        # index_to_reduce -> i vector
        index_to_reduce = [self.variables.index(var) for var in var_to_reduce]

        mu_j = self.mean[index_to_keep]
        mu_i = self.mean[index_to_reduce]
        x_i = np.array([value for var, value in values]).reshape(len(index_to_reduce), 1)

        sig_i_j = self.covariance[np.ix_(index_to_reduce, index_to_keep)]
        sig_j_i = self.covariance[np.ix_(index_to_keep, index_to_reduce)]
        sig_i_i_inv = np.linalg.inv(self.covariance[np.ix_(index_to_reduce, index_to_reduce)])
        sig_j_j = self.covariance[np.ix_(index_to_keep, index_to_keep)]

        phi.variables = [self.variables[index] for index in index_to_keep]
        phi.mean = mu_j + np.dot(np.dot(sig_j_i, sig_i_i_inv), x_i - mu_i)
        phi.covariance = sig_j_j - np.dot(np.dot(sig_j_i, sig_i_i_inv), sig_i_j)
        phi._precision_matrix = None

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        """
        Normalizes the distribution. In case of a Gaussian Distribution the
        distribution is always normalized, therefore this method doesn't do
        anything and has been implemented only for a consistent API across
        distributions.
        """
        phi = self if inplace else self.copy()

        # The pdf of a Joint Gaussian distrinution is always
        # normalized. Hence, no changes.
        if not inplace:
            return phi

    def copy(self):
        """
        Return a copy of the distribution.

        Returns
        -------
        GaussianDistribution: copy of the distribution

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> gauss_dis = GD(variables=['x1', 'x2', 'x3'],
        ...                mean=[1, -3, 4],
        ...                cov=[[4, 2, -2],
        ...                     [2, 5, -5],
        ...                     [-2, -5, 8]])
        >>> copy_dis = gauss_dis.copy()
        >>> copy_dis.variables
        ['x1', 'x2', 'x3']
        >>> copy_dis.mean
        array([[ 1],
                [-3],
                [ 4]])
        >>> copy_dis.covariance
        array([[ 4,  2, -2],
                [ 2,  5, -5],
                [-2, -5,  8]])
        >>> copy_dis.precision_matrix
        array([[ 0.3125    , -0.125     ,  0.        ],
                [-0.125     ,  0.58333333,  0.33333333],
                [ 0.        ,  0.33333333,  0.33333333]])
        """
        copy_distribution = GaussianDistribution(variables=self.variables,
                                                 mean=self.mean.copy(),
                                                 cov=self.covariance.copy())
        if self._precision_matrix is not None:
            copy_distribution._precision_matrix = self._precision_matrix.copy()

        return copy_distribution

    def to_canonical_factor(self):
        u"""
        Returns an equivalent CanonicalDistribution object.

        The formulas for calculating the cannonical factor parameters
        for N(μ; Σ) = C(K; h; g) are as follows -

        K = sigma^(-1)
        h = sigma^(-1) * mu
        g = -(0.5) * mu.T * sigma^(-1) * mu -
            log((2*pi)^(n/2) * det(sigma)^(0.5))

        where,
        K,h,g are the canonical factor parameters
        sigma is the covariance_matrix of the distribution,
        mu is the mean_vector of the distribution,
        mu.T is the transpose of the matrix mu,
        and det(sigma) is the determinant of the matrix sigma.

        Example
        -------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis = GD(variables=['x1', 'x2', 'x3'],
        ...          mean=[1, -3, 4],
        ...          cov=[[4, 2, -2],
        ...               [2, 5, -5],
        ...               [-2, -5, 8]])
        >>> phi = dis.to_canonical_factor()
        >>> phi.variables
        ['x1', 'x2', 'x3']
        >>> phi.K
        array([[0.3125, -0.125, 0.],
               [-0.125, 0.5833, 0.333],
               [     0., 0.333, 0.333]])
        >>> phi.h
        array([[  0.6875],
               [-0.54166],
               [ 0.33333]]))
        >>> phi.g
        -6.51533
        """
        from pgmpy.factors.continuous import CanonicalDistribution

        mu = self.mean
        sigma = self.covariance

        K = self.precision_matrix

        h = np.dot(K, mu)

        g = -(0.5) * np.dot(mu.T, h)[0, 0] - np.log(
            np.power(2 * np.pi, len(self.variables)/2) *
            np.power(abs(np.linalg.det(sigma)), 0.5))

        return CanonicalDistribution(self.variables, K, h, g)

    def _operate(self, other, operation, inplace=True):
        """
        Gives the CanonicalDistribution operation (product or divide) with
        the other factor.

        Parameters
        ----------
        other: CanonicalDistribution
            The CanonicalDistribution to be multiplied.

        operation: String
            'product' for multiplication operation and
            'divide' for division operation.

        Returns
        -------
        CanonicalDistribution or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new CanonicalDistribution instance.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis1 = GD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...             np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        >>> dis2 = GD(['x3', 'x4'], [1, 2], [[2, 3], [5, 6]])
        >>> dis3 = dis1 * dis2
        >>> dis3.covariance
        array([[ 3.6,  1. , -0.4, -0.6],
               [ 1. ,  2.5, -1. , -1.5],
               [-0.4, -1. ,  1.6,  2.4],
               [-1. , -2.5,  4. ,  4.5]])
        >>> dis3.mean
        array([[ 1.6],
               [-1.5],
               [ 1.6],
               [ 3.5]])
        """
        phi = self.to_canonical_factor()._operate(
            other.to_canonical_factor(), operation, inplace=False).to_joint_gaussian()

        if not inplace:
            return phi

    def product(self, other, inplace=True):
        """
        TODO: Make it work when using `*` instead of product.

        Returns the product of two gaussian distributions.

        Parameters
        ----------
        other: GaussianDistribution
            The GaussianDistribution to be multiplied.

        inplace: boolean
            If True, modifies the distribution itself, otherwise returns a new
            GaussianDistribution object.

        Returns
        -------
        CanonicalDistribution or None:
                    if inplace=True (default) returns None.
                    if inplace=False returns a new CanonicalDistribution instance.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis1 = GD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...            np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        >>> dis2 = GD(['x3', 'x4'], [1, 2], [[2, 3], [5, 6]])
        >>> dis3 = dis1.product(dis2, inplace=False)
        >>> dis3.covariance
        array([[ 3.6,  1. , -0.4, -0.6],
               [ 1. ,  2.5, -1. , -1.5],
               [-0.4, -1. ,  1.6,  2.4],
               [-1. , -2.5,  4. ,  4.5]])
        >>> dis3.mean
        array([[ 1.6],
               [-1.5],
               [ 1.6],
               [ 3.5]])
        """
        return self._operate(other, operation='product', inplace=inplace)

    def divide(self, other, inplace=True):
        """
        Returns the division of two gaussian distributions.

        Parameters
        ----------
        other: GaussianDistribution
            The GaussianDistribution to be divided.

        inplace: boolean
            If True, modifies the distribution itself, otherwise returns a new
            GaussianDistribution object.

        Returns
        -------
        CanonicalDistribution or None:
                    if inplace=True (default) returns None.
                    if inplace=False returns a new CanonicalDistribution instance.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.distributions import GaussianDistribution as GD
        >>> dis1 = GD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...            np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        >>> dis2 = GD(['x3', 'x4'], [1, 2], [[2, 3], [5, 6]])
        >>> dis3 = dis1.divide(dis2, inplace=False)
        >>> dis3.covariance
        array([[ 3.6,  1. , -0.4, -0.6],
               [ 1. ,  2.5, -1. , -1.5],
               [-0.4, -1. ,  1.6,  2.4],
               [-1. , -2.5,  4. ,  4.5]])
        >>> dis3.mean
        array([[ 1.6],
               [-1.5],
               [ 1.6],
               [ 3.5]])
        """
        return self._operate(other, operation='divide', inplace=inplace)

    def __repr__(self):
        return "GaussianDistribution representing N({var}) at {address}".format(
            var=self.variables, address=hex(id(self)))

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__

    def __eq__(self, other):
        if not (isinstance(self, GaussianDistribution) and isinstance(self, GaussianDistribution)):
            return False

        elif set(self.scope()) != set(other.scope()):
            return False

        else:
            # Computing transform_index to be able to easily have variables in same order.
            transform_index = [other.index(var) for var in self.variables]

            if not np.allclose(self.mean, other.mean[transform_index]):
                return False
            else:
                mid_cov = other.covariance[transform_index, :]
                transform_cov = mid_cov[:, transform_index]
                if not np.allclose(self.covariance, transform_cov):
                    return False
        return True
