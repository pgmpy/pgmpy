# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from pgmpy.extern import six
from pgmpy.factors.base import BaseFactor
from pgmpy.factors.distributions import GaussianDistribution


class LinearGaussianCPD(BaseFactor):
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
    def __init__(self, variable, mean, cov, evidence=[], dist=None):
        """
        Initialize a new LinearGaussianCPD object.

        Parameters
        ----------
        variable: Any hashable python object (optional)
            The variable on which the CPD is defined. If dist is an instance of `BaseDistribution`,
            the variable would automatically be taken from dist.


        Examples
        --------

        """
        if dist:
            super(LinearGaussianCPD, self).__init__(dist=dist)
        else:
            variables = [variable]
            if len(mean) != len(evidence) + 1:
                raise ValueError("mean: Expected size: {ex}, got: {size}".format(
                    ex=len(evidence)+1, size=len(mean)))

            super(LinearGaussianCPD, self).__init__(variables=variables,
                                                    mean=mean,
                                                    cov=cov,
                                                    evidence=evidence)

