import types

import numpy as np
import scipy.integrate as integrate

from pgmpy.factors.base import BaseFactor
from pgmpy.factors.distributions import GaussianDistribution, CustomDistribution


class ContinuousFactor(BaseFactor):
    """
    Base class for factors representing various multivariate
    representations.
    """

    def __init__(self, variables, pdf, *args, **kwargs):
        """
        Parameters
        ----------
        variables: list or array-like
            The variables for which the distribution is defined.

        pdf: function
            The probability density function of the distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.continuous import ContinuousFactor
        # Two variable dirichlet distribution with alpha = (1,2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_factor = ContinuousFactor(['x', 'y'], dirichlet_pdf)
        >>> dirichlet_factor.scope()
        ['x', 'y']
        >>> dirichlet_factor.assignment(5,6)
        226800.0
        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError(
                f"variables: Expected type list or array-like, got type {type(variables)}"
            )

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same.")

        variables = list(variables)

        if isinstance(pdf, str):
            if pdf == "gaussian":
                self.distribution = GaussianDistribution(
                    variables=variables,
                    mean=kwargs["mean"],
                    covariance=kwargs["covariance"],
                )
            else:
                raise NotImplementedError(
                    f"{pdf} distribution not supported. Please use CustomDistribution"
                )

        elif isinstance(pdf, CustomDistribution):
            self.distribution = pdf

        elif callable(pdf):
            self.distribution = CustomDistribution(
                variables=variables, distribution=pdf
            )

        else:
            raise ValueError(
                f"pdf: Expected type: str or function, Got: {type(variables)}"
            )

    @property
    def pdf(self):
        """
        Returns the pdf of the ContinuousFactor.
        """
        return self.distribution.pdf

    @property
    def variable(self):
        return self.scope()[0]

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
        return self.distribution.variables

    def get_evidence(self):
        return self.scope()[1:]

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
        return self.distribution.assignment(*args)

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
        # Two variable dirichlet distribution with alpha = (1,2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_factor = ContinuousFactor(['x', 'y'], dirichlet_pdf)
        >>> dirichlet_factor.variables
        ['x', 'y']
        >>> copy_factor = dirichlet_factor.copy()
        >>> copy_factor.variables
        ['x', 'y']
        """
        return ContinuousFactor(self.scope(), self.distribution.copy())

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
            a new ContinuousFactor object.

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
        phi = self if inplace else self.copy()

        phi.distribution = phi.distribution.reduce(values, inplace=False)
        if not inplace:
            return phi

    def marginalize(self, variables, inplace=True):
        """
        Marginalize the factor with respect to the given variables.

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
        phi = self if inplace else self.copy()
        phi.distribution = phi.distribution.marginalize(variables, inplace=False)

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
        phi.distribution.normalize(inplace=True)

        if not inplace:
            return phi

    def is_valid_cpd(self):
        return self.distribution.is_valid_cpd()

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
            raise TypeError(
                f"ContinuousFactor objects can only be multiplied ",
                f"or divided with another ContinuousFactor object. ",
                f"Got {type(other)}, expected: ContinuousFactor.",
            )

        phi = self if inplace else self.copy()
        phi.distribution = phi.distribution._operate(
            other=other.distribution, operation=operation, inplace=False
        )

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
                        if inplace=False returns a new `ContinuousFactor` instance.

        Examples
        --------
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
        return self._operate(other, "product", inplace)

    def divide(self, other, inplace=True):
        """
        Gives the ContinuousFactor divide with the other factor.

        Parameters
        ----------
        other: ContinuousFactor
            The ContinuousFactor to be divided.

        Returns
        -------
        ContinuousFactor or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `ContinuousFactor` instance.

        Examples
        --------
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
        if set(other.scope()) - set(self.scope()):
            raise ValueError("Scope of divisor should be a subset of dividend")

        return self._operate(other, "divide", inplace)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__
