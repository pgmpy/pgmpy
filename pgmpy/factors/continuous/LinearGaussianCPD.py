# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal

from pgmpy.extern import six
from pgmpy.factors import ContinuousFactor


class LinearGaussianCPD(ContinuousFactor):
    u"""
    For, X -> Y the Linear Gaussian model assumes that the mean
    of Y is a liner function of X and that the variance of Y does
    not depend on X.

    For example,
    p(Y|X) = N(-2x + 0.9 ; 1)

    Let Y be a continuous variable with continuous parents
    X1 ............ Xk . We say that Y has a linear Gaussian CPD
    if there are parameters β0,.........βk and σ2 such that,

    p(Y |x1.......xk) = N(β0 + β1 + ......... + βk ; σ2)

    In vector notation,

    p(Y |x) = N(β0 + β.T * x ; σ2)

    """
    def __init__(self, variable, beta_not, variance, evidence=None, beta_vector=None):
        """
        Parameters
        ----------

        variable: any hashable python object
            The variable whose CPD is defined.

        beta_not: int, float
            Represents the constant term in the linear equation.

        variance: int, float
            The variance of the variable defined.

        evidence: iterable of any hashabale python objects
            An iterable of the parents of the variable. None
            if there are no parents.

        beta_vector: iterable of int or float
            An iterable representing the coefficient vector of the linear equation.

        Examples
        --------

        # For P(Y| X1, X2, X3) = N(-2x1 + 3x2 + 7x3 + 0.2; 9.6)

        >>> cpd = LinearGaussianCPD('Y', 0.2, 9.6, ['X1', 'X2', 'X3'], [-2, 3, 7])
        >>> cpd.variable
        'Y'
        >>> cpd.variance
        9.6
        >>> cpd.evidence
        ['x1', 'x2', 'x3']
        >>> cpd.beta_vector
        [-2, 3, 7]
        >>> cpd.beta_not
        0.2

        """
        self.variable = variable
        self.beta_not = beta_not
        self.variance = variance

        if len(evidence) != len(beta_vector):
            raise ValueError("The number of variables in evidence must be equal to the "
                             "length of the beta vector.")

        self.evidence = evidence
        self.beta_vector = beta_vector

        variables = [variable] + evidence
        super(LinearGaussianCPD, self).__init__(variables, None)

