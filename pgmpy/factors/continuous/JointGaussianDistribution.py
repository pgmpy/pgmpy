from __future__ import division

import six
import numpy as np
from scipy.stats import multivariate_normal

from pgmpy.factors import ContinuousFactor


class JointGaussianDistribution(ContinuousFactor):
    """
    In its most common representation, a multivariate Gaussian distribution
    over X1...........Xn is characterized by an n-dimensional mean vector μ,
    and a symmetric n x n covariance matrix Σ.
    This is the base class for its representation.
    """
    def __init__(self, variables, mean, covariance):
        """
        Parameters
        ----------
        variables: iterable of any hashable python object
            The variables for which the distribution is defined.
        mean: n x 1, array like 
            n-dimensional vector where n is the number of variables.
        covariance: n x n, 2-d array like
            n x n dimensional matrix where n is the number of variables.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> dis = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...             np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        >>> dis.variables
        ['x1', 'x2', 'x3']
        >>> dis.mean
        array([[ 1],
               [-3],
               [4]]))
        >>> dis.covariance
        matrix([[4, 2, -2],
               [2, 5, -5],
               [-2, -5, 8]])
        >>> dis.pdf([0,0,0])
        0.0014805631279234139
        """
        # dimension of the mean vector and covariance matrix
        no_of_var = len(variables)

        if len(mean) != no_of_var:
            raise ValueError("Length of mean_vector must be equal to the\
                                 number of variables.")

        mean = np.asarray(np.reshape(mean, (no_of_var, 1)), dtype=float)

        covariance = np.asarray(covariance, dtype=float)

        if covariance.shape != (no_of_var, no_of_var):
            raise ValueError("The Covariance matrix should be a square matrix with order equal to\
                              the number of variables. Got: {got_shape}, Expected: {exp_shape}"
                              .format(got_shape=covariance.shape,
                              exp_shape=(no_of_var, no_of_var)))

        normal_pdf = lambda x: multivariate_normal.pdf(x, mean.reshape(1,no_of_var)[0], covariance)

        super(JointGaussianDistribution, self).__init__(variables,
                                                        normal_pdf, mean, covariance)

        self._precision_matrix = None

    @property
    def precision_matrix(self):
        """
        Returns the precision matrix of the distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> dis = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...             np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
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

        variables: iterator
                List of variables over which to marginalize.
        inplace: boolean
                If inplace=True it will modify the distribution itself,
                else would return a new distribution.

        Returns
        -------
        JointGaussianDistribution or None :
                if inplace=True (default) returns None
                if inplace=False return a new JointGaussianDistribution instance

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> dis = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...             np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
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
        array([[ 1],
                [-3]]))
        >>> dis.covariance
        narray([[4, 2],
               [2, 5]])
        """

        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        distribution = self if inplace else self.copy()

        var_indexes = [distribution.variables.index(var) for var in variables]
        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))

        distribution.variables = [distribution.variables[index] for index in index_to_keep]
        distribution.mean = distribution.mean[index_to_keep]
        distribution.covariance = distribution.covariance[np.ix_(index_to_keep, index_to_keep)]
        distribution._precision_matrix = None

        if not inplace:
            return distribution

    def copy(self):
        """
        Return a copy of the distribution.

        Returns
        -------
        JointGaussianDistribution: copy of the distribution

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> gauss_dis = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...                 np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
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
        copy_distribution = JointGaussianDistribution(self.variables.copy(), self.mean.copy(),
                                                      self.covariance.copy())
        if self._precision_matrix is not None:
            copy_distribution._precision_matrix = self._precision_matrix.copy()
        else:
            copy_distribution._precision_matrix = None

        return copy_distribution

    def to_canonical_factor(self):
        """
        Returns an equivalent CanonicalFactor object.

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

        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> dis = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]]),
        ...             np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        >>> phi = dis.toCanonicalFactor()
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
        # TODO: This method will return a CanonicalFactor object.
        # Currently this cannot be used until the CanonicalFactor class is
        # created. For now the method returns a tuple of the CanonicalFactor
        # parameters - (K, h, g)
        mu = self.mean
        sigma = self.covariance

        K = self.precision_matrix

        h = np.dot(K, mu)

        g = -(0.5) * np.dot(mu.T, h)[0, 0] - np.log(np.power
                    (2 * np.pi, len(self.variables)/2) * np.power(np.linalg.det(sigma), 0.5))

        return K,h,g
