from __future__ import division

from pgmpy.factors.base import BaseFactor

import numpy as np
import scipy.integrate as integrate


class ContinuousFactor(BaseFactor):
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
        >>> from pgmpy.factors.continuous import ContinuousFactor
        # Two variable drichlet ditribution with alpha = (1,2)
        >>> def drichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_factor = ContinuousFactor(['x', 'y'], drichlet_pdf)
        >>> dirichlet_factor.scope()
        ['x', 'y']
        >>> dirichlet_factor.assignemnt(5,6)
        226800.0
        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError("variables: Expected type list or array-like, "
                            "got type {var_type}".format(var_type=type(variables)))

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same.")

        self.variables = list(variables)
        self._pdf = pdf

    @property
    def pdf(self):
        """
        Returns the pdf of the ContinuousFactor.
        """
        return self._pdf

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
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
        *args: values
            Values whose assignment is to be computed.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf = lambda x1, x2: multivariate_normal.pdf((x1, x2), [0, 0], [[1, 0], [0, 1]])
        >>> phi = ContinuousFactor(['x1', 'x2'], normal_pdf)
        >>> phi.assignment(1, 2)
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
        >>> from pgmpy.factors.continuous import ContinuousFactor
        # Two variable drichlet ditribution with alpha = (1,2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_factor = ContinuousFactor(['x', 'y'], dirichlet_pdf)
        >>> dirichlet_factor.variables
        ['x', 'y']
        >>> copy_factor = dirichlet_factor.copy()
        >>> copy_factor.variables
        ['x', 'y']
        """
        return ContinuousFactor(self.scope(), self.pdf)

    def discretize(self, method, *args, **kwargs):
        """
        Discretizes the continuous distribution into discrete
        probability masses using various methods.

        Parameters
        ----------
        method : A Discretizer Class from pgmpy.discretize

        *args, **kwargs:
            The parameters to be given to the Discretizer Class.

        Returns
        -------
        An n-D array or a DiscreteFactor object according to the discretiztion
        method used.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from pgmpy.factors.continuous import RoundingDiscretizer
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_factor = ContinuousFactor(['x', 'y'], dirichlet_pdf)
        >>> dirichlet_factor.discretize(RoundingDiscretizer, low=1, high=2, cardinality=5)
        # TODO: finish this
        """
        return method(self, *args, **kwargs).get_discrete_values()

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
                                  if inplace=False returns a new ContinuousFactor instance.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> def custom_pdf(x, y, z):
        ...     return z*(np.power(x, 1) * np.power(y, 2)) / beta(x, y)
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
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("variables: Expected type list or array-like, "
                            "got type {var_type}".format(var_type=type(values)))

        for var, value in values:
            if var not in self.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        phi = self if inplace else self.copy()

        var_to_remove = [var for var, value in values]
        var_to_keep = [var for var in self.variables if var not in var_to_remove]

        reduced_var_index = [(self.variables.index(var), value) for var, value in values]
        pdf = self.pdf

        def reduced_pdf(*args, **kwargs):
            reduced_args = list(args)
            reduced_kwargs = kwargs.copy()

            if reduced_args:
                for index, val in reduced_var_index:
                    reduced_args.insert(index, val)
            if reduced_kwargs:
                for variable, val in values:
                    reduced_kwargs[variable] = val
            if reduced_args and reduced_kwargs:
                reduced_args = [arg for arg in reduced_args if arg not in reduced_kwargs.values()]

            return pdf(*reduced_args, **reduced_kwargs)

        phi.variables = var_to_keep
        phi._pdf = reduced_pdf

        if not inplace:
            return phi

    def marginalize(self, variables, inplace=True):
        """
        Maximizes the factor with respect to the given variables.

        Parameters
        ----------
        variables: list, array-like
            List of variables with respect to which factor is to be maximized.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new ContinuousFactor instance.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new ContinuousFactor instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> std_normal_pdf = lambda *x: multivariate_normal.pdf(x, [0, 0], [[1, 0], [0, 1]])
        >>> std_normal = ContinuousFactor(['x1', 'x2'], std_normal_pdf)
        >>> std_normal.scope()
        ['x1', 'x2']
        >>> std_normal.assignment([1, 1])
        0.058549831524319168
        >>> std_normal.marginalize(['x2'])
        >>> std_normal.scope()
        ['x1']
        >>> std_normal.assignment(1)

        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError("variables: Expected type list or array-like, "
                            "got type {var_type}".format(var_type=type(variables)))

        for var in variables:
            if var not in self.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        phi = self if inplace else self.copy()

        all_var = [var for var in self.variables]
        var_to_keep = [var for var in self.variables if var not in variables]
        reordered_var_index = [all_var.index(var) for var in variables + var_to_keep]
        pdf = phi.pdf

        # The arguments need to be reordered because integrate.nquad integrates the first n-arguments
        # of the function passed.

        def reordered_pdf(*args):
            # ordered_args restores the original order as it was in self.variables
            ordered_args = [args[reordered_var_index.index(index_id)] for index_id in range(len(all_var))]
            return pdf(*ordered_args)

        def marginalized_pdf(*args):
            return integrate.nquad(reordered_pdf, [[-np.inf, np.inf] for i in range(len(variables))], args=args)[0]

        phi._pdf = marginalized_pdf
        phi.variables = var_to_keep

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        """
        Normalizes the pdf of the continuous factor so that it integrates to
        1 over all the variables.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        ContinuousFactor or None:
             if inplace=True (default) returns None
             if inplace=False returns a new ContinuousFactor instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> std_normal_pdf = lambda x: 2 * multivariate_normal.pdf(x, [0, 0], [[1, 0], [0, 1]])
        >>> std_normal = ContinuousFactor(['x1', 'x2'], std_normal_pdf)
        >>> std_normal.assignment(1, 1)
        0.117099663049
        >>> std_normal.normalize()
        >>> std_normal.assignment(1, 1)
        0.0585498315243

        """
        phi = self if inplace else self.copy()
        pdf = self.pdf

        pdf_mod = integrate.nquad(pdf, [[-np.inf, np.inf] for var in self.variables])[0]

        phi._pdf = lambda *args: pdf(*args) / pdf_mod

        if not inplace:
            return phi

    def _operate(self, other, operation, inplace=True):
        """
        Gives the ContinuousFactor operation (product or divide) with
        the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be multiplied.

        operation: String
            'product' for multiplication operation and 'divide' for
            division operation.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        """
        if not isinstance(other, ContinuousFactor):
            raise TypeError("ContinuousFactor object can only be multiplied or divided with "
                            "an another ContinuousFactor object. Got {other_type}, expected "
                            "ContinuousFactor.".format(other_type=type(other)))

        phi = self if inplace else self.copy()
        pdf = self.pdf
        self_var = [var for var in self.variables]

        modified_pdf_var = self_var + [var for var in other.variables if var not in self_var]

        def modified_pdf(*args):
            self_pdf_args = list(args[:len(self_var)])
            other_pdf_args = [args[modified_pdf_var.index(var)] for var in other.variables]

            if operation == 'product':
                return pdf(*self_pdf_args) * other.pdf(*other_pdf_args)
            if operation == 'divide':
                return pdf(*self_pdf_args) / other.pdf(*other_pdf_args)

        phi.variables = modified_pdf_var
        phi._pdf = modified_pdf

        if not inplace:
            return phi

    def product(self, other, inplace=True):
        """
        Gives the ContinuousFactor product with the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be multiplied.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Example
        -------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> sn_pdf1 = lambda x: multivariate_normal.pdf([x], [0], [[1]])
        >>> sn_pdf2 = lambda x1,x2: multivariate_normal.pdf([x1, x2], [0, 0], [[1, 0], [0, 1]])
        >>> sn1 = ContinuousFactor(['x2'], sn_pdf1)
        >>> sn2 = ContinuousFactor(['x1', 'x2'], sn_pdf2)

        >>> sn3 = sn1.product(sn2, inplace=False)
        >>> sn3.assignment(0, 0)
        0.063493635934240983

        >>> sn3 = sn1 * sn2
        >>> sn3.assignment(0, 0)
        0.063493635934240983
        """
        return self._operate(other, 'product', inplace)

    def divide(self, other, inplace=True):
        """
        Gives the ContinuousFactor divide with the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be multiplied.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Example
        -------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from scipy.stats import multivariate_normal
        >>> sn_pdf1 = lambda x: multivariate_normal.pdf([x], [0], [[1]])
        >>> sn_pdf2 = lambda x1,x2: multivariate_normal.pdf([x1, x2], [0, 0], [[1, 0], [0, 1]])
        >>> sn1 = ContinuousFactor(['x2'], sn_pdf1)
        >>> sn2 = ContinuousFactor(['x1', 'x2'], sn_pdf2)

        >>> sn4 = sn2.divide(sn1, inplace=False)
        >>> sn4.assignment(0, 0)
        0.3989422804014327

        >>> sn4 = sn2 / sn1
        >>> sn4.assignment(0, 0)
        0.3989422804014327
        """
        if set(other.variables) - set(self.variables):
            raise ValueError("Scope of divisor should be a subset of dividend")

        return self._operate(other, 'divide', inplace)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__
