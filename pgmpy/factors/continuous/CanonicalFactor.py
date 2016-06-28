# -*- coding: utf-8 -*-

from __future__ import division

import six
import numpy as np
from scipy.stats import multivariate_normal

from pgmpy.factors import ContinuousFactor
from pgmpy.factors import JointGaussianDistribution

class CanonicalFactor(ContinuousFactor):
    u"""
    The intermediate factors in a Gaussian network can be described
    compactly using a simple parametric representation called the
    canonical form. This representation is closed under the basic
    operations used in inference: factor product, factor division,
    factor reduction, and marginalization. Thus, we define this
    CanonicalFactor class that allows the inference process to be
    performed. We can represent every Gaussian as a canonical form.

    A canonical form C (X; K,h,g) is defined as

    C (X; K,h,g) = exp( ((-1/2) * X.T * K * X) + (h.T * X) + g)

    Reference
    ---------
    Probabilistic Graphical Models, Principles and Techniques,
    Daphne Koller and Nir Friedman, Section 14.2, Chapter 14.

    """
    def __init__(self, variables, K, h, g, pdf=None):
        """
        Parameters
        ----------
        Parameters
        ----------
        variables: list or array-like
        The variables for wich the distribution is defined.

        K: n x n, 2-d array-like

        h : n x 1, array-like

        g : int, float

        The terms K, h and g are defined parameters for canonical
        factors representation.

        pdf: function
        The probability density function of the distribution.

        Examples
        --------
        Examples
        --------
        >>> from pgmpy.factors import CanonicalFactor
        >>> phi = CanonicalFactor(['X', 'Y'], np.array([[1, -1], [-1, 1]]),
                                  np.array([[1], [-1]]), -3)
        >>> phi.variables
        ['X', 'Y']

        >>> phi.K
        matrix([[1, -1],
               [-1, 1]])

        >>> phi.h
        matrix([[1],
               [-1]])

        >>> phi.g
        -3

        """
        no_of_var = len(variables)

        if len(h) != no_of_var:
            raise ValueError("Length of h parameter vector must be equal to the"
                         "number of variables.")

        self.h = np.asarray(np.reshape(h, (no_of_var, 1)), dtype=float)
        self.g = g
        self.K = np.asarray(K, dtype=float)

        if self.K.shape != (no_of_var, no_of_var):
            raise ValueError("The K matrix should be a square matrix with order equal to"
                             "the number of variables. Got: {got_shape}, Expected: {exp_shape}".format
                             (got_shape=self.covariance.shape, exp_shape=(no_of_var, no_of_var)))

        super(CanonicalFactor, self).__init__(variables, pdf)

    def assignment(self, *args):
        if self.pdf is None:
            self.pdf = self.to_joint_gaussian().pdf
        return super(CanonicalFactor, self).assignment(*args)

    def copy(self):
        """
        Makes a copy of the factor.

        Returns
        -------
        CanonicalFactor object: Copy of the factor

        Examples
        --------
        >>> from pgmpy.factors import CanonicalFactor
        >>> phi = CanonicalFactor(['X', 'Y'], np.array([[1, -1], [-1, 1]]),
                                  np.array([[1], [-1]]), -3)
        >>> phi.variables
        ['X', 'Y']

        >>> phi.K
        matrix([[1, -1],
               [-1, 1]])

        >>> phi.h
        matrix([[1],
               [-1]])

        >>> phi.g
        -3

        >>> phi2 = phi.copy()

        >>> phi2.variables
        ['X', 'Y']

        >>> phi2.K
        matrix([[1, -1],
               [-1, 1]])

        >>> phi2.h
        matrix([[1],
               [-1]])

        >>> phi2.g
        -3

        """
        copy_factor = CanonicalFactor(self.scope(), self.K.copy(),
                                      self.h.copy(), self.g, self.pdf)
        if self.pdf is not None:
            copy_factor.pdf = self.pdf

        return copy_factor

    def to_joint_gaussian(self):
        """
        Return an equivalent Joint Gaussian Distribution.

        Examples
        --------

        >>> import numpy as np
        >>> from pgmpy.factors import CanonicalFactor
        >>> phi = CanonicalFactor(['x1', 'x2'], np.array([[3, -2], [-2, 4]]),
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

        return JointGaussianDistribution(self.scope(), mean, covariance)
