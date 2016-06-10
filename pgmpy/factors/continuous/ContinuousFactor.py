import six


class ContinuousFactor(object):
    """
    Base class for factors representing various multivariate
    representations.
    """
    def __init__(self, variables, pdf, mean=None, covariance=None, lb=None, ub=None):
        """
        Parameters
        ----------
        variables: list or array-like
            The variables for wich the distribution is defined.
        pdf: function
            The probability density function of the distribution.
        mean: list or array-like
            The mean of the distribution.
        covariance: A 2-D array or matrix
            The covariance of the distribution.
        lb: list or tuple of floats or integers
            Lower bound of the support of the distribution, for each
            variable.
        ub: list or tuple of floats or integers
            Upper bound of the support of the distribution, for each
            variable.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        # Two variable drichlet ditribution with alpha = (1,2)
        >>> def drichlet_pdf(x, y):
        ...     return (np.power(x, 1)*np.power(y, 2))/beta(x, y)
        >>> from pgmpy.factors import ContinuousFactor
        >>> drichlet_factor = ContinuousFactor(['x', 'y'], drichlet_pdf)
        >>> drichlet_factor.scope()
        ['x', 'y']
        >>> drichlet_factor.assignemnt(5,6)
        226800.0
        """
        if isinstance(variables, six.string_types):
            raise TypeError("Variables: Expected type list or array like, got string")

        self.variables = variables
        self.pdf = pdf
        self.mean = mean
        self.covariance = covariance
        self.lb = lb
        self.ub = ub

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf = lambda x: multivariate_normal(x, [0, 0], [[1, 0], [0, 1]])
        >>> phi = ContinuousFactor(['x1', 'x2'], normal_pdf)
        >>> phi.scope()
        ['x1', 'x2']
        """
        return self.variables

    def assignment(self, *args):
        """
        Returns a list of pdf assignments for the corresponding values.
        
        Parameters
        ----------
        values: A list of arrays of dimension 1 x n
            List of values whose assignment is to be computed.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf = lambda x: multivariate_normal.pdf(x, [0, 0], [[1, 0], [0, 1]])
        >>> phi = ContinuousFactor(['x1', 'x2'], normal_pdf)
        >>> phi.assignment([1,2])
        0.013064233284684921
        """
        return self.pdf(*args)


    def discretize(self, method, *args, **kwargs):
        """
        Discretizes the continuous distribution into discrete
        probability masses using various methods.

        Returns
        -------
        An n-D array or a Factor object according to the discretiztion
        method used.
 
        Parameters
        ----------
        method : A Discretizer Class from pgmpy.continuous.discretize
 
        *args, **kwargs:
            The parameters to be given to the Discretizer Class.
        """
        return method(self, *args, **kwargs).get_discrete_values()
