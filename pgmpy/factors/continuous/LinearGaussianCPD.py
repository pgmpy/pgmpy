# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from pgmpy.extern import six
from pgmpy.factors.base import BaseFactor
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.distributions import GaussianDistribution


class LinearGaussianCPD(ContinuousFactor):
    u"""
    For, X -> Y the Linear Gaussian model assumes that the mean
    of Y is a linear function of mean of X and the variance of Y does
    not depend on X.

    For example,
    p(Y|X) = N(-2x + 0.9 ; 1)
    Here, x is the mean of the variable X.

    Let Y be a continuous variable with continuous parents
    X1 ............ Xk . We say that Y has a linear Gaussian CPD
    if there are parameters β0,.........βk and σ2 such that,

    p(Y |x1.......xk) = N(β0 + x1*β1 + ......... + xk*βk ; σ2)

    In vector notation,

    p(Y |x) = N(β0 + β.T * x ; σ2)

    """
    def __init__(self, variable=None, mean=None, cov=None, evidence=[], dist=None):
        """
        Initialize a new LinearGaussianCPD object.

        Parameters
        ----------
        variable: Any hashable python object (optional)
            The variable on which the CPD is defined.
            Not required if dist is not None.

        mean: list/iterable (optional)
            The mean of the Gaussian Distribution. For a Linear Conditional Gaussian 
            Distribution `P(Y | A, B, C) = N(2A + 3B - 4C + 1; 1)`, the mean vector
            should be [2, 3, -4].
            Not required if `dist` is not None.

        cov: float (optional)
            The covariance of the `varaible`.
            Not reuired if `dist` is not None.

        evidence: list/iterable (optional)
            The variables on which the LinearGaussianCPD is conditioned on. 
            Not required if `dist` is not None or there are no conditional variables.

        dist: Instance of pgmpy.factors.distributions.GaussianDistribution (optional)
            The GaussianDistribution which is to be represented by the LinearGaussianCPD
            instance. The GaussianDistribution should have only one variable.
            Not required if all other parameters are given.

        Examples
        --------
        For representing the following CPD:
            P(Y | A, B, C) = N(2A + 3B - 4C + 1; 1)

        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> cpd = LinearGaussianCPD(variable='Y', mean=[2, 3, -4, 1], cov=1)
        """
        if dist:
            if len(dist.variables) != 1:
                raise ValueError("The distribution should be defined on a single variable")
            super(LinearGaussianCPD, self).__init__(dist=dist)
        else:
            variables = [variable]
            if len(mean) != len(evidence) + 1:
                raise ValueError("mean: Expected size: {ex}, got: {size}".format(
                    ex=len(evidence)+1, size=len(mean)))
            
            dist = GaussianDistribution(variables=variables,
                                        mean=mean,
                                        cov=cov,
                                        evidence=evidence)
                                        
            super(LinearGaussianCPD, self).__init__(dist=dist)
