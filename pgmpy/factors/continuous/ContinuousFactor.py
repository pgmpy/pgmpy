from __future__ import division

from pgmpy.factors.base import BaseFactor
from pgmpy.factors.distributions import CustomDistribution

class ContinuousFactor(BaseFactor):
    """
    Base class for factors representing various multivariate
    representations.

    for variable elimination:
    We need to implement marginalize,reduction,normalize


    """
    def __init__(self, distribution, variable=None):
        """
        Parameters
        ----------
        distribution: CustomDistribution
            The distribution of variables in this factor.
        variable: str
            The name of the dependent variable if this factor represents a CPD

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from pgmpy.factors.distributions import CustomDistribution
        # Two variable drichlet ditribution with alpha = (1,2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'],distribution=dirichlet_pdf)
        >>> dirichlet_factor = ContinuousFactor(dirichlet_dist)
        >>> dirichlet_factor.scope
        ['x', 'y']
        >>> dirichlet_factor.assignment(5,6)
        226800.0
        """
        if distribution and not isinstance(distribution, CustomDistribution):
            raise TypeError(
                "distribution: Expected type: CustomDistribution, got: {type}".format(
                    type=type(distribution)))

        self.distribution = distribution
        self.variable = variable # The variable to predict on
        super().__init__()

    @property
    def evidence(self):
        return self.scope[1:]

    @property
    def pdf(self):
        """
        Returns the pdf of the ContinuousFactor.
        """
        return self.distribution.pdf

    @property
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

    def assignment(self, *x):
        """
        Returns a list of pdf assignments for the corresponding values.

        Parameters
        ----------
        *x: values
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
        return self.pdf(*x)

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
        return ContinuousFactor(self.distribution.copy())

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
        distribution = self.distribution.reduce(values, inplace)
        if not inplace:
            s = self.copy()
            s.distribution = distribution
            return s

    def marginalize(self, variables, inplace=True):
        distribution = self.distribution.marginalize(variables, inplace)
        if not inplace:
            s = self.copy()
            s.distribution = distribution
            return s

    def normalize(self, inplace=True):
        distribution = self.distribution.normalize(inplace)
        if not inplace:
            s = self.copy()
            s.distribution = distribution
            return s

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
        self_var = [var for var in self.scope]

        modified_pdf_var = self_var + [var for var in other.scope if var not in self_var]

        def modified_pdf(*args):
            self_pdf_args = list(args[:len(self_var)])
            other_pdf_args = [args[modified_pdf_var.index(var)] for var in other.scope]

            if operation == 'product':
                return pdf(*self_pdf_args) * other.pdf(*other_pdf_args)
            if operation == 'divide':
                return pdf(*self_pdf_args) / other.pdf(*other_pdf_args)

        phi.distribution = CustomDistribution(modified_pdf_var, modified_pdf)

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
                        if inplace=False returns a new `Continuous` instance.

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
        if set(other.scope) - set(self.scope):
            raise ValueError("Scope of divisor should be a subset of dividend")

        return self._operate(other, 'divide', inplace)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__
