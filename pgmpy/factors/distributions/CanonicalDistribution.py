# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from pgmpy.factors.distributions import BaseDistribution
from pgmpy.factors.distributions import GaussianDistribution


class CanonicalDistribution(BaseDistribution):
    u"""
    The intermediate factors in a Gaussian network can be described
    compactly using a simple parametric representation called the
    canonical form. This representation is closed under the basic
    operations used in inference: factor product, factor division,
    factor reduction, and marginalization. Thus, we define this
    CanonicalDistribution class that allows the inference process to be
    performed on joint Gaussian networks.

    A canonical form C (X; K,h,g) is defined as

    C (X; K,h,g) = exp( ((-1/2) * X.T * K * X) + (h.T * X) + g)

    References
    ----------
    Probabilistic Graphical Models, Principles and Techniques,
    Daphne Koller and Nir Friedman, Section 14.2, Chapter 14.

    """

    def __init__(self, variables, K, h, g):
        """
        Parameters
        ----------
        variables: list or array-like
        The variables for wich the distribution is defined.

        K: n x n, 2-d array-like

        h : n x 1, array-like

        g : int, float

        pdf: function
        The probability density function of the distribution.

        The terms K, h and g are defined parameters for canonical
        factors representation.

        Examples
        --------
        >>> from pgmpy.factors.continuous import CanonicalDistribution
        >>> phi = CanonicalDistribution(['X', 'Y'], np.array([[1, -1], [-1, 1]]),
                                  np.array([[1], [-1]]), -3)
        >>> phi.variables
        ['X', 'Y']

        >>> phi.K
        array([[1, -1],
               [-1, 1]])

        >>> phi.h
        array([[1],
               [-1]])

        >>> phi.g
        -3

        """
        no_of_var = len(variables)

        if len(h) != no_of_var:
            raise ValueError(
                "Length of h parameter vector must be equal to "
                "the number of variables."
            )

        self.variables = variables
        self.h = np.asarray(np.reshape(h, (no_of_var, 1)), dtype=float)
        self.g = g
        self.K = np.asarray(K, dtype=float)

        if self.K.shape != (no_of_var, no_of_var):
            raise ValueError(
                "The K matrix should be a square matrix with "
                "order equal to the number of variables. Got: "
                "{got_shape}, Expected: {exp_shape}".format(
                    got_shape=self.K.shape, exp_shape=(no_of_var, no_of_var)
                )
            )

    @property
    def pdf(self):
        def fun(*args):
            x = np.array(args)
            return np.exp(
                self.g + np.dot(x, self.h)[0] - 0.5 * np.dot(x.T, np.dot(self.K, x))
            )

        return fun

    def assignment(self, *x):
        """
        Returns the probability value of the PDF at the given parameter values.
        Parameters
        ----------
        *x: values of all variables of this distribution,
            collective defining a point at which the probability value is to be computed.
        Returns
        -------
        float: The probability value at the point.
        Examples
        --------
        >>> from pgmpy.factors.distributions import GaussianDistribution
        >>> dist = GaussianDistribution(variables=['x1', 'x2'],
        ...                             mean=[[0], [0]],
        ...                             cov=[[1, 0], [0, 1]])
        >>> dist.assignment(0, 0)
        0.15915494309189535
        """
        return self.pdf(*x)

    def copy(self):
        """
        Makes a copy of the factor.

        Returns
        -------
        CanonicalDistribution object: Copy of the factor

        Examples
        --------
        >>> from pgmpy.factors.continuous import CanonicalDistribution
        >>> phi = CanonicalDistribution(['X', 'Y'], np.array([[1, -1], [-1, 1]]),
                                  np.array([[1], [-1]]), -3)
        >>> phi.variables
        ['X', 'Y']

        >>> phi.K
        array([[1, -1],
               [-1, 1]])

        >>> phi.h
        array([[1],
               [-1]])

        >>> phi.g
        -3

        >>> phi2 = phi.copy()

        >>> phi2.variables
        ['X', 'Y']

        >>> phi2.K
        array([[1, -1],
               [-1, 1]])

        >>> phi2.h
        array([[1],
               [-1]])

        >>> phi2.g
        -3

        """
        copy_factor = CanonicalDistribution(
            self.variables, self.K.copy(), self.h.copy(), self.g
        )

        return copy_factor

    def to_joint_gaussian(self):
        """
        Return an equivalent Joint Gaussian Distribution.

        Examples
        --------

        >>> import numpy as np
        >>> from pgmpy.factors.continuous import CanonicalDistribution
        >>> phi = CanonicalDistribution(['x1', 'x2'], np.array([[3, -2], [-2, 4]]),
                                  np.array([[5], [-1]]), 1)
        >>> jgd = phi.to_joint_gaussian()
        >>> jgd.variables
        ['x1', 'x2']
        >>> jgd.covariance
        array([[ 0.5  ,  0.25 ],
               [ 0.25 ,  0.375]])
        >>> jgd.mean
        array([[ 2.25 ],
               [ 0.875]])

        """
        covariance = np.linalg.inv(self.K)
        mean = np.dot(covariance, self.h)

        return GaussianDistribution(self.variables, mean, covariance)

    def reduce(self, values, inplace=True):
        """
        Reduces the distribution to the context of the given variable values.

        Let C(X,Y ; K, h, g) be some canonical form over X,Y where,

        k = [[K_XX, K_XY],        ;       h = [[h_X],
             [K_YX, K_YY]]                     [h_Y]]

        The formula for the obtained conditional distribution for setting
        Y = y is given by,

        .. math:: K' = K_{XX}
        .. math:: h' = h_X - K_{XY} * y
        .. math:: g' = g + {h^T}_Y * y - 0.5 * y^T * K_{YY} * y


        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable name, variable value).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new CanonicalFactor object.

        Returns
        -------
        CanonicalDistribution or None:
                if inplace=True (default) returns None
                if inplace=False returns a new CanonicalDistribution instance.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.continuous import CanonicalDistribution
        >>> phi = CanonicalDistribution(['X1', 'X2', 'X3'],
        ...                       np.array([[1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
        ...                       np.array([[1], [4], [-1]]), -2)
        >>> phi.variables
        ['X1', 'X2', 'X3']

        >>> phi.K
        array([[ 1., -1.],
               [-1.,  3.]])

        >>> phi.h
        array([[ 1. ],
               [ 3.5]])

        >>> phi.g
        -2

        >>> phi.reduce([('X3', 0.25)])

        >>> phi.variables
        ['X1', 'X2']

        >>> phi.K
        array([[ 1, -1],
               [-1,  4]])

        >>> phi.h
        array([[ 1. ],
               [ 4.5]])

        >>> phi.g
        -2.375
        """
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type list or array-like, "
                "got type {var_type}".format(var_type=type(values))
            )

        if not all([var in self.variables for var, value in values]):
            raise ValueError("Variable not in scope.")

        phi = self if inplace else self.copy()

        var_to_reduce = [var for var, value in values]

        # index_to_keep -> j vector
        index_to_keep = [
            self.variables.index(var)
            for var in self.variables
            if var not in var_to_reduce
        ]
        # index_to_reduce -> i vector
        index_to_reduce = [self.variables.index(var) for var in var_to_reduce]

        K_i_i = self.K[np.ix_(index_to_keep, index_to_keep)]
        K_i_j = self.K[np.ix_(index_to_keep, index_to_reduce)]
        K_j_j = self.K[np.ix_(index_to_reduce, index_to_reduce)]
        h_i = self.h[index_to_keep]
        h_j = self.h[index_to_reduce]

        # The values for the reduced variables.
        y = np.array([value for var, value in values]).reshape(len(index_to_reduce), 1)

        phi.variables = [self.variables[index] for index in index_to_keep]
        phi.K = K_i_i
        phi.h = h_i - np.dot(K_i_j, y)
        phi.g = (
            self.g + (np.dot(h_j.T, y) - (0.5 * np.dot(np.dot(y.T, K_j_j), y)))[0][0]
        )

        if not inplace:
            return phi

    def marginalize(self, variables, inplace=True):
        u"""
        Modifies the factor with marginalized values.

        Let C(X,Y ; K, h, g) be some canonical form over X,Y where,

        k = [[K_XX, K_XY],        ;       h = [[h_X],
             [K_YX, K_YY]]                     [h_Y]]

        In this case, the result of the integration operation is a canonical
        from C (K', h', g') given by,

        .. math:: K' = K_{XX} - K_{XY} * {K^{-1}}_{YY} * K_YX
        .. math:: h' = h_X - K_{XY} * {K^{-1}}_{YY} * h_Y
        .. math:: g' = g + 0.5 * (|Y| * log(2*pi) - log(|K_{YY}|) + {h^T}_Y * K_{YY} * h_Y)

        Parameters
        ----------

        variables: list or array-like
                List of variables over which to marginalize.

        inplace: boolean
                If inplace=True it will modify the distribution itself,
                else would return a new distribution.

        Returns
        -------
        CanonicalDistribution or None :
                if inplace=True (default) returns None
                if inplace=False return a new CanonicalDistribution instance

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.continuous import CanonicalDistribution
        >>> phi = CanonicalDistribution(['X1', 'X2', 'X3'],
        ...                       np.array([[1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
        ...                       np.array([[1], [4], [-1]]), -2)
        >>> phi.K
        array([[ 1, -1,  0],
                [-1,  4, -2],
                [ 0, -2,  4]])

        >>> phi.h
        array([[ 1],
                [ 4],
                [-1]])

        >>> phi.g
        -2

        >>> phi.marginalize(['X3'])

        >>> phi.K
        array([[ 1., -1.],
                [-1.,  3.]])

        >>> phi.h
        array([[ 1. ],
                [ 3.5]])

        >>> phi.g
        0.22579135
        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type list or array-like, "
                "got type {var_type}".format(var_type=type(variables))
            )

        if not all([var in self.variables for var in variables]):
            raise ValueError("Variable not in scope.")

        phi = self if inplace else self.copy()

        # index_to_keep -> i vector
        index_to_keep = [
            self.variables.index(var) for var in self.variables if var not in variables
        ]
        # index_to_marginalize -> j vector
        index_to_marginalize = [self.variables.index(var) for var in variables]

        K_i_i = self.K[np.ix_(index_to_keep, index_to_keep)]
        K_i_j = self.K[np.ix_(index_to_keep, index_to_marginalize)]
        K_j_i = self.K[np.ix_(index_to_marginalize, index_to_keep)]
        K_j_j = self.K[np.ix_(index_to_marginalize, index_to_marginalize)]
        K_j_j_inv = np.linalg.inv(K_j_j)
        h_i = self.h[index_to_keep]
        h_j = self.h[index_to_marginalize]

        phi.variables = [self.variables[index] for index in index_to_keep]

        phi.K = K_i_i - np.dot(np.dot(K_i_j, K_j_j_inv), K_j_i)
        phi.h = h_i - np.dot(np.dot(K_i_j, K_j_j_inv), h_j)
        phi.g = (
            self.g
            + 0.5
            * (
                len(variables) * np.log(2 * np.pi)
                - np.log(abs(np.linalg.det(K_j_j)))
                + np.dot(np.dot(h_j.T, K_j_j), h_j)
            )[0][0]
        )

        if not inplace:
            return phi

    def _operate(self, other, operation, inplace=True):
        """
        Gives the CanonicalDistribution operation (product or divide) with
        the other factor.

        The product of two canonical factors over the same scope
        X is simply:

        C(K1, h1, g1) * C(K2, h2, g2) = C(K1+K2, h1+h2, g1+g2)

        The division of canonical forms is defined analogously:

        C(K1, h1, g1) / C(K2, h2, g2) = C(K1-K2, h1-h2, g1- g2)

        When we have two canonical factors over different scopes X and Y,
        we simply extend the scope of both to make their scopes match and
        then perform the operation of the above equation. The extension of
        the scope is performed by simply adding zero entries to both the K
        matrices and the h vectors.

        Parameters
        ----------
        other: CanonicalFactor
            The CanonicalDistribution to be multiplied.

        operation: String
            'product' for multiplication operation and
            'divide' for division operation.

        Returns
        -------
        CanonicalDistribution or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new CanonicalDistribution instance.

        Example
        -------
        >>> import numpy as np
        >>> from pgmpy.factors.continuous import CanonicalDistribution
        >>> phi1 = CanonicalDistribution(['x1', 'x2', 'x3'],
                                   np.array([[1, -1, 0], [-1, 4, -2], [0, -2, 4]]),
                                   np.array([[1], [4], [-1]]), -2)
        >>> phi2 = CanonicalDistribution(['x1', 'x2'], np.array([[3, -2], [-2, 4]]),
                                   np.array([[5], [-1]]), 1)

        >>> phi3 = phi1 * phi2
        >>> phi3.K
        array([[ 4., -3.,  0.],
               [-3.,  8., -2.],
               [ 0., -2.,  4.]])
        >>> phi3.h
        array([ 6.,  3., -1.])
        >>> phi3.g
        -1

        >>> phi4 = phi1 / phi2
        >>> phi4.K
        array([[-2.,  1.,  0.],
               [ 1.,  0., -2.],
               [ 0., -2.,  4.]])
        >>> phi4.h
        array([-4.,  5., -1.])
        >>> phi4.g
        -3

        """
        if not isinstance(other, CanonicalDistribution):
            raise TypeError(
                "CanonicalDistribution object can only be multiplied or divided "
                "with an another CanonicalDistribution object. Got {other_type}, "
                "expected CanonicalDistribution.".format(other_type=type(other))
            )

        phi = self if inplace else self.copy()

        all_vars = self.variables + [
            var for var in other.variables if var not in self.variables
        ]
        no_of_var = len(all_vars)

        self_var_index = [all_vars.index(var) for var in self.variables]
        other_var_index = [all_vars.index(var) for var in other.variables]

        def _extend_K_scope(K, index):
            ext_K = np.zeros([no_of_var, no_of_var])
            ext_K[np.ix_(index, index)] = K
            return ext_K

        def _extend_h_scope(h, index):
            ext_h = np.zeros(no_of_var).reshape(no_of_var, 1)
            ext_h[index] = h
            return ext_h

        phi.variables = all_vars

        if operation == "product":
            phi.K = _extend_K_scope(self.K, self_var_index) + _extend_K_scope(
                other.K, other_var_index
            )
            phi.h = _extend_h_scope(self.h, self_var_index) + _extend_h_scope(
                other.h, other_var_index
            )
            phi.g = self.g + other.g

        else:
            phi.K = _extend_K_scope(self.K, self_var_index) - _extend_K_scope(
                other.K, other_var_index
            )
            phi.h = _extend_h_scope(self.h, self_var_index) - _extend_h_scope(
                other.h, other_var_index
            )
            phi.g = self.g - other.g

        if not inplace:
            return phi

    def product(self, other, inplace=True):
        """
        Returns the product of two gaussian distributions.

        Parameters
        ----------
        other: CanonicalFactor
            The GaussianDistribution to be multiplied.

        inplace: boolean
            If True, modifies the distribution itself, otherwise returns a new
            CanonicalDistribution object.

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
        return self._operate(other, operation="product", inplace=inplace)

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
        return self._operate(other, operation="divide", inplace=inplace)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__
