# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pgmpy.factors.base import BaseFactor


class LinearGaussianCPD(BaseFactor):
    r"""
    Defines a Linear Gaussian CPD.

    The Linear Gaussian CPD makes the following assumptions:
        1) The variable is Gaussian/Normally distributed.
        2) The mean of the variable depends on the values of the parents and the
            intercept term.
        3) The variance is independent of other variables.

    For example,

    .. math::

      p(Y|X) = N(0.9 - 2x; 1)

    Here, :math:`0.9 - 2x` is the mean of the variable :math:`Y` and the
    standard deviation is 1.

    In generalized terms, let :math:`Y` be a Gaussian variable with parents
    :math:`X_1, X_2, \cdots, X_k`. Assuming linear relationship between Y and
    \mathbf{X}, the conditional distribution of Y can be defined as:

    .. math:: p(Y |x1, x2, ..., xk) = \mathcal{N}(\beta_0 + x1*\beta_1 + ......... + xk*\beta_k ; \sigma)

    References
    ----------
    .. [1] https://cedar.buffalo.edu/~srihari/CSE574/Chap8/Ch8-PGM-GaussianBNs/8.5%20GaussianBNs.pdf

    Parameters
    ----------

    variable: any hashable python object
        The variable whose CPD is defined.

    beta: list (array-like)
        The coefficients corresponding to each of the evidence variable. The first
        term of the `beta` array is the intercept term.

    std: float
        The standard deviation of `variable`.

    evidence: iterator (array-like)
        List of parents/evidence variables of `variable`. The order in which `evidence`
        is specified should match the order of `beta`.

    Examples
    --------
    # To represent the conditional distribution, P(Y| X1, X2, X3) = N(0.2 - 2*x1 + 3*x2 + 7*x3 ; 9.6), we can write:

    >>> from pgmpy.factors.continuous import LinearGaussianCPD
    >>> cpd = LinearGaussianCPD('Y',  [0.2, -2, 3, 7], 9.6, ['X1', 'X2', 'X3'])
    >>> cpd.variable
    'Y'
    >>> cpd.evidence
    ['x1', 'x2', 'x3']
    >>> cpd.beta_vector
    [0.2, -2, 3, 7]
    """

    def __init__(self, variable, beta, std, evidence=[]):
        self.variable = variable
        self.beta = np.array(beta)
        self.std = std
        self.evidence = list(evidence)
        self.variables = [variable] + evidence

    def copy(self):
        """
        Returns a copy of the distribution.

        Returns
        -------
        LinearGaussianCPD: copy of the distribution

        Examples
        --------
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> cpd = LinearGaussianCPD('Y',  [0.2, -2, 3, 7], 9.6, ['X1', 'X2', 'X3'])
        >>> copy_cpd = cpd.copy()
        >>> copy_cpd.variable
        'Y'
        >>> copy_cpd.evidence
        ['X1', 'X2', 'X3']
        """
        copy_cpd = LinearGaussianCPD(
            variable=self.variable,
            beta=self.beta,
            std=self.std,
            evidence=list(self.evidence),
        )

        return copy_cpd

    def __str__(self):
        mean = self.beta.round(3)
        std = round(self.std, 3)
        if self.evidence and list(self.beta):
            # P(Y| X1, X2, X3) = N(-2*X1_mu + 3*X2_mu + 7*X3_mu; 0.2)
            rep_str = "P({node} | {parents}) = N({mu} + {b_0}; {sigma})".format(
                node=str(self.variable),
                parents=", ".join([str(var) for var in self.evidence]),
                mu=" + ".join(
                    [
                        f"{coeff}*{parent}"
                        for coeff, parent in zip(mean[1:], self.evidence)
                    ]
                ),
                b_0=str(mean[0]),
                sigma=str(std),
            )
        else:
            # P(X) = N(1, 4)
            rep_str = f"P({str(self.variable)}) = N({str(mean[0])}; {str(std)})"
        return rep_str

    def __repr__(self):
        str_repr = self.__str__()
        return f"<LinearGaussianCPD: {str_repr} at {hex(id(self))}"
