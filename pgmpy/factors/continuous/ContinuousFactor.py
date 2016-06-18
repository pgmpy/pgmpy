import six
import numpy as np
import scipy.integrate as integrate


class ContinuousFactor(object):
    """
    Base class for factors representing various multivariate
    representations.
    """
    def __init__(self, variables, pdf):
        """
        Parameters
        ----------
        variables: list or array-like
            The variables for wich the distribution is defined.

        pdf: function
            The probability density function of the distribution.

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

    def copy(self):
        """
        Return a copy of the distribution.

        Returns
        -------
        ContinuousFactor object: copy of the distribution

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        # Two variable drichlet ditribution with alpha = (1,2)
        >>> def drichlet_pdf(x, y):
        ...     return (np.power(x, 1)*np.power(y, 2))/beta(x, y)
        >>> from pgmpy.factors import ContinuousFactor
        >>> drichlet_factor = ContinuousFactor(['x', 'y'], drichlet_pdf)
        >>> drichlet_factor.variables
        ['x', 'y']
        >>> copy_factor = drichlet_factor.copy()
        copy_factor.variables

        """
        return ContinuousFactor(self.scope(), self.pdf)

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of the given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_value).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new ContinuosFactor object.

        Returns
        -------
        ContinuousFactor or None: if inplace=True (default) returns None
                                  if inplace=False returns a new `Factor` instance.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> def custom_pdf(x, y, z):
        ...     return z*(np.power(x, 1)*np.power(y, 2))/beta(x, y)
        >>> from pgmpy.factors import ContinuousFactor
        >>> custom_factor = ContinuousFactor(['x', 'y', 'z'], custom_pdf)
        >>> custom_factor.variables
        ['x', 'y', 'z']
        >>> custom_factor.assignment(1, 2, 3)
        24.0

        >>> custom_factor.reduce([('y', 2)])
        >>> custom_factor.variables
        ['x', 'z']
        >>> custom_factor.assignment(1, 3)
        24.0
        """
        if isinstance(values, six.string_types):
            raise TypeError("values: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        var_to_remove = [var for var, value in values]
        var_to_keep = [var for var in self.variables if var not in var_to_remove]

        reduced_var_index = [(self.variables.index(var), value) for var, value in values]
        pdf = self.pdf

        def reduced_pdf(*args, **kwargs):
            reduced_args = list(args)
            reduced_kwargs = kwargs.copy()

            if reduced_args:
                for index, value in reduced_var_index:
                    reduced_args.insert(index, value)
            if reduced_kwargs:
                for var, value in values:
                    reduced_kwargs[var] = value
            if reduced_args and reduced_kwargs:
                reduced_args = [arg for arg in reduced_args if arg not in reduced_kwargs.values()]

            return pdf(*reduced_args, **reduced_kwargs)

        phi.variables = var_to_keep
        phi.pdf = reduced_pdf

        if not inplace:
            return phi

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
