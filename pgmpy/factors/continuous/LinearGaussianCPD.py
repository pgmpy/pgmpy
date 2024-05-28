# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pgmpy.factors.base import BaseFactor


class LinearGaussianCPD(BaseFactor):
    r"""
    For, X -> Y the Linear Gaussian model assumes that the mean
    of Y is a linear function of mean of X and the variance of Y does
    not depend on X.

    For example,

    .. math::

      p(Y|X) = N(-2x + 0.9 ; 1)

    Here, :math:`x` is the mean of the variable :math:`X`.

    Let :math:`Y` be a continuous variable with continuous parents
    :math:`X1, X2, \cdots, Xk`. We say that :math:`Y` has a linear Gaussian CPD
    if there are parameters :math:`\beta_0, \beta_1, ..., \beta_k`
    and :math:`\sigma_2` such that,

    .. math:: p(Y |x1, x2, ..., xk) = \mathcal{N}(\beta_0 + x1*\beta_1 + ......... + xk*\beta_k ; \sigma_2)

    In vector notation,

    .. math:: p(Y |x) = \mathcal{N}(\beta_0 + \boldmath{β}.T * \boldmath{x} ; \sigma_2)

    References
    ----------
    .. [1] https://cedar.buffalo.edu/~srihari/CSE574/Chap8/Ch8-PGM-GaussianBNs/8.5%20GaussianBNs.pdf
    """

    def __init__(
        self, variable, evidence_mean, evidence_variance, evidence=[], beta=None
    ):
        """
        Parameters
        ----------

        variable: any hashable python object
            The variable whose CPD is defined.

        evidence_mean: Mean vector (numpy array) of the joint distribution, X

        evidence_variance: int, float
            The variance of the multivariate gaussian, X = ['x1', 'x2', ..., 'xn']

        evidence: iterable of any hashable python objects
            An iterable of the parents of the variable. None if there are no parents.

        beta (optional): iterable of int or float
            An iterable representing the coefficient vector of the linear equation.
            The first term represents the constant term in the linear equation.


        Examples
        --------

        # For P(Y| X1, X2, X3) = N(-2x1 + 3x2 + 7x3 + 0.2; 9.6)

        >>> cpd = LinearGaussianCPD('Y',  [0.2, -2, 3, 7], 9.6, ['X1', 'X2', 'X3'])
        >>> cpd.variable
        'Y'
        >>> cpd.evidence
        ['x1', 'x2', 'x3']
        >>> cpd.beta_vector
        [0.2, -2, 3, 7]

        """
        self.variable = variable
        self.mean = np.array(evidence_mean)
        self.variance = evidence_variance
        self.evidence = evidence
        self.sigma_yx = None

        self.variables = [variable] + evidence
        super(LinearGaussianCPD, self).__init__(
            self.variables, pdf="gaussian", mean=self.mean, covariance=self.variance
        )

    def sum_of_product(self, xi, xj):
        prod_xixj = xi * xj
        return np.sum(prod_xixj)

    def maximum_likelihood_estimator(self, data, states):
        """
        Fit using MLE method.

        Parameters
        ----------
        data: pandas.DataFrame or 2D array
            Dataframe of values containing samples from the conditional distribution, (Y|X)
            and corresponding X values.

        states: All the input states that are jointly gaussian.

        Returns
        -------
        beta, variance (tuple): Returns estimated betas and the variance.
        """
        x_df = pd.DataFrame(data, columns=states)
        x_len = len(self.evidence)

        sym_coefs = []
        for i in range(0, x_len):
            sym_coefs.append("b" + str(i + 1) + "_coef")

        sum_x = x_df.sum()
        x = [sum_x["(Y|X)"]]
        coef_matrix = pd.DataFrame(columns=sym_coefs)

        # First we compute just the coefficients of beta_1 to beta_N.
        # Later we compute beta_0 and append it.
        for i in range(0, x_len):
            x.append(self.sum_of_product(x_df["(Y|X)"], x_df[self.evidence[i]]))
            for j in range(0, x_len):
                coef_matrix.loc[i, sym_coefs[j]] = self.sum_of_product(
                    x_df[self.evidence[i]], x_df[self.evidence[j]]
                )

        coef_matrix.insert(0, "b0_coef", sum_x[self.evidence].values)
        row_1 = np.append([len(x_df)], sum_x[self.evidence].values)
        coef_matrix.loc[-1] = row_1
        coef_matrix.index = coef_matrix.index + 1  # shifting index
        coef_matrix.sort_index(inplace=True)

        beta_coef_matrix = np.matrix(coef_matrix.values, dtype="float")
        coef_inv = np.linalg.inv(beta_coef_matrix)
        beta_est = np.array(np.matmul(coef_inv, np.transpose(x)))
        self.beta = beta_est[0]

        sigma_est = 0
        x_len_df = len(x_df)
        for i in range(0, x_len):
            for j in range(0, x_len):
                sigma_est += (
                    self.beta[i + 1]
                    * self.beta[j + 1]
                    * (
                        self.sum_of_product(
                            x_df[self.evidence[i]], x_df[self.evidence[j]]
                        )
                        / x_len_df
                        - np.mean(x_df[self.evidence[i]])
                        * np.mean(x_df[self.evidence[j]])
                    )
                )

        sigma_est = np.sqrt(
            self.sum_of_product(x_df["(Y|X)"], x_df["(Y|X)"]) / x_len_df
            - np.mean(x_df["(Y|X)"]) * np.mean(x_df["(Y|X)"])
            - sigma_est
        )
        self.sigma_yx = sigma_est
        return self.beta, self.sigma_yx

    def fit(self, data, states, estimator=None, **kwargs):
        """
        Determine βs from data

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing samples from the conditional distribution, p(Y|X)
            estimator: 'MLE' or 'MAP'
        """
        if estimator == "MLE":
            mean, variance = self.maximum_likelihood_estimator(data, states)
        elif estimator == "MAP":
            raise NotImplementedError(
                "fit method has not been implemented using Maximum A-Priori (MAP)"
            )

        return mean, variance

    @property
    def pdf(self):
        def _pdf(*args):
            # The first element of args is the value of the variable on which CPD is defined
            # and the rest of the elements give the mean values of the parent
            # variables.
            mean = (
                sum([arg * coeff for (arg, coeff) in zip(args[1:], self.mean)])
                + self.mean[0]
            )
            return multivariate_normal.pdf(
                args[0], np.array(mean), np.array([[self.variance]])
            )

        return _pdf

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
            self.variable, self.beta, self.variance, list(self.evidence)
        )

        return copy_cpd

    def __str__(self):
        mean = self.mean.round(3)
        variance = round(self.variance, 3)
        if self.evidence and list(self.mean):
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
                sigma=str(variance),
            )
        else:
            # P(X) = N(1, 4)
            rep_str = f"P({str(self.variable)}) = N({str(mean[0])}; {str(variance)})"
        return rep_str

    def __repr__(self):
        str_repr = self.__str__()
        return f"<LinearGaussianCPD: {str_repr} at {hex(id(self))}"
