import numpy as np
from scipy import integrate

from pgmpy.factors.distributions import BaseDistribution


class CustomDistribution(BaseDistribution):
    def __init__(self, variables, distribution, *args, **kwargs):
        """
        Class for representing custom continuous distributions.

        Parameters
        ----------
        variables: list or array-like
            The variables for which the distribution is defined.

        distribution: function
            The probability density function of the distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.distributions import CustomDistribution
        # Two variable dirichlet distribution with alpha = (1, 2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'], distribution=dirichlet_pdf)
        >>> dirichlet_dist.variables
        ['x', 'y']
        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type: iterable, got: {type}".format(
                    type=type(variables)
                )
            )

        if len(set(variables)) != len(variables):
            raise ValueError("Multiple variables can't have the same name")

        self._variables = list(variables)
        self._pdf = distribution

    @property
    def pdf(self):
        """
        Returns the Probability Density Function of the distribution.

        Returns
        -------
        function: The probability density function of the distribution

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.distributions import CustomDistribution
        # Two variable dirichlet distribution with alpha = (1, 2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'],
        ...                                     distribution=dirichlet_pdf)
        >>> dirichlet_dist.pdf()
        <function __main__.diri_pdf>
        """
        return self._pdf

    @pdf.setter
    def pdf(self, f):
        self._pdf = f

    @property
    def variables(self):
        """
        Returns the scope of the distribution.

        Returns
        -------
        list: List of variables on which the distribution is defined.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.distributions import CustomDistribution
        # Two variable dirichlet distribution with alpha = (1, 2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'],
        ...                                     distribution=dirichlet_pdf)
        >>> dirichlet_dist.variables
        ['x', 'y']
        """
        return self._variables

    @variables.setter
    def variables(self, value):
        self._variables = value

    def assignment(self, *x):
        """
        Returns the probability value of the PDF at the given parameter values.

        Parameters
        ----------
        *x: values of all variables of this distribution,
            collective defining a point at which the probability value is to be computed.

        Returns
        -------
        float: The probability value at the point.

        Examples
        --------
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf = lambda x1, x2: multivariate_normal.pdf(
        ...     x=(x1, x2), mean=[0, 0], cov=[[1, 0], [0, 1]])
        >>> normal_dist = CustomDistribution(variables=['x1', 'x2'],
        ...                                  distribution=normal_pdf)
        >>> normal_dist.assignment(0, 0)
        0.15915494309189535
        0.15915494309189535
        """
        return self.pdf(*x)

    def copy(self):
        """
        Returns a copy of the CustomDistribution instance.

        Returns
        -------
        CustomDistribution object: copy of the instance

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.distributions import CustomDistribution
        # Two variable dirichlet distribution with alpha = (1,2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'],
        ...                                     distribution=dirichlet_pdf)
        >>> copy_dist = dirichlet_dist.copy()
        >>> copy_dist.variables
        ['x', 'y']
        """
        return CustomDistribution(self.variables, self.pdf)

    # TODO: Discretize methods need to be fixed for this to work
    def discretize(self, method, *args, **kwargs):
        """
        Discretizes the continuous distribution into discrete
        probability masses using specified method.

        Parameters
        ----------
        method: string, BaseDiscretizer instance
            A Discretizer Class from pgmpy.factors.discretize

        *args, **kwargs: values
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
        >>> dirichlet_factor.discretize(RoundingDiscretizer,
        ...                             low=1, high=2, cardinality=5)
        # TODO: finish this
        """
        super(CustomDistribution, self).discretize(method, *args, **kwargs)

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of the given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_value).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new CustomDistribution object.

        Returns
        -------
        CustomDistribution or None:
                    if inplace=True (default) returns None
                    if inplace=False returns a new CustomDistribution instance.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> def custom_pdf(x, y, z):
        ...     return z*(np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> custom_dist = CustomDistribution(['x', 'y', 'z'], custom_pdf)
        >>> custom_dist.variables
        ['x', 'y', 'z']
        >>> custom_dist.assignment(1, 2, 3)
        24.0

        >>> custom_dist.reduce([('y', 2)])
        >>> custom_dist.variables
        ['x', 'z']
        >>> custom_dist.assignment(1, 3)
        24.0
        """
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type: iterable, "
                "got: {var_type}".format(var_type=type(values))
            )

        for var, value in values:
            if var not in self.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        phi = self if inplace else self.copy()

        var_to_remove = [var for var, value in values]
        var_to_keep = [var for var in self.variables if var not in var_to_remove]

        reduced_var_index = [
            (self.variables.index(var), value) for var, value in values
        ]
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
                reduced_args = [
                    arg for arg in reduced_args if arg not in reduced_kwargs.values()
                ]

            return pdf(*reduced_args, **reduced_kwargs)

        phi.variables = var_to_keep
        phi._pdf = reduced_pdf

        if not inplace:
            return phi

    def marginalize(self, variables, inplace=True):
        """
        Marginalize the distribution with respect to the given variables.

        Parameters
        ----------
        variables: list, array-like
            List of variables to be removed from the marginalized distribution.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new CustomDistribution instance.

        Returns
        -------
        Marginalized distribution or None:
                if inplace=True (default) returns None
                if inplace=False returns a new CustomDistribution instance.

        Examples
        --------
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf = lambda x1, x2: multivariate_normal.pdf(
        ...                x=[x1, x2], mean=[0, 0], cov=[[1, 0], [0, 1]])
        >>> normal_dist = CustomDistribution(variables=['x1', 'x2'],
        ...                                  distribution=normal_pdf)
        >>> normal_dist.variables
        ['x1', 'x2']
        >>> normal_dist.assignment(1, 1)
        0.058549831524319168
        >>> normal_dist.marginalize(['x2'])
        >>> normal_dist.variables
        ['x1']
        >>> normal_dist.assignment(1)
        0.24197072451914328
        """
        if len(variables) == 0:
            raise ValueError("Shouldn't be calling marginalize over no variable.")

        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type iterable, "
                "got: {var_type}".format(var_type=type(variables))
            )

        for var in variables:
            if var not in self.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        phi = self if inplace else self.copy()

        all_var = [var for var in self.variables]
        var_to_keep = [var for var in self.variables if var not in variables]
        reordered_var_index = [all_var.index(var) for var in variables + var_to_keep]
        pdf = phi._pdf

        # The arguments need to be reordered because integrate.nquad
        # integrates the first n-arguments of the function passed.

        def reordered_pdf(*args):
            # ordered_args restores the original order as it was in self.variables
            ordered_args = [
                args[reordered_var_index.index(index_id)]
                for index_id in range(len(all_var))
            ]
            return pdf(*ordered_args)

        def marginalized_pdf(*args):
            return integrate.nquad(
                reordered_pdf,
                [[-np.inf, np.inf] for i in range(len(variables))],
                args=args,
            )[0]

        phi._pdf = marginalized_pdf
        phi.variables = var_to_keep

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        """
        Normalizes the pdf of the distribution so that it
        integrates to 1 over all the variables.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the distribution itself, else would return
            a new distribution.

        Returns
        -------
        CustomDistribution or None:
             if inplace=True (default) returns None
             if inplace=False returns a new CustomDistribution instance.

        Examples
        --------
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf_x2 = lambda x1, x2: 2 * multivariate_normal.pdf(
        ...                     x=[x1, x2], mean=[0, 0], cov=[[1, 0], [0, 1]])
        >>> normal_dist_x2 = CustomDistribution(variables=['x1', 'x2'],
        ...                                     distribution=normal_pdf_x2)
        >>> normal_dist_x2.assignment(1, 1)
        0.117099663049
        >>> normal_dist = normal_dist_x2.normalize(inplace=False)
        >>> normal_dist.assignment(1, 1)
        0.0585498315243
        """
        phi = self if inplace else self.copy()
        pdf = self.pdf

        pdf_mod = integrate.nquad(pdf, [[-np.inf, np.inf] for var in self.variables])[0]

        phi._pdf = lambda *args: pdf(*args) / pdf_mod

        if not inplace:
            return phi

    def is_valid_cpd(self):
        return np.isclose(
            integrate.nquad(self.pdf, [[-np.inf, np.inf] for var in self.variables])[0],
            1,
        )

    def _operate(self, other, operation, inplace=True):
        """
        Gives the CustomDistribution operation (product or divide) with
        the other distribution.

        Parameters
        ----------
        other: CustomDistribution
            The CustomDistribution to be multiplied.

        operation: str
            'product' for multiplication operation and 'divide' for
            division operation.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new distribution.

        Returns
        -------
        CustomDistribution or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `CustomDistribution` instance.

        """
        if not isinstance(other, CustomDistribution):
            raise TypeError(
                "CustomDistribution objects can only be multiplied "
                "or divided with another CustomDistribution  "
                "object. Got {other_type}, expected: "
                "CustomDistribution.".format(other_type=type(other))
            )

        phi = self if inplace else self.copy()
        pdf = self.pdf
        self_var = [var for var in self.variables]

        modified_pdf_var = self_var + [
            var for var in other.variables if var not in self_var
        ]

        def modified_pdf(*args):
            self_pdf_args = list(args[: len(self_var)])
            other_pdf_args = [
                args[modified_pdf_var.index(var)] for var in other.variables
            ]

            if operation == "product":
                return pdf(*self_pdf_args) * other._pdf(*other_pdf_args)
            if operation == "divide":
                return pdf(*self_pdf_args) / other._pdf(*other_pdf_args)

        phi.variables = modified_pdf_var
        phi._pdf = modified_pdf

        if not inplace:
            return phi

    def product(self, other, inplace=True):
        """
        Gives the CustomDistribution product with the other distribution.

        Parameters
        ----------
        other: CustomDistribution
            The CustomDistribution to be multiplied.

        Returns
        -------
        CustomDistribution or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `CustomDistribution` instance.

        Example
        -------
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> from scipy.stats import multivariate_normal
        >>> sn_pdf1 = lambda x: multivariate_normal.pdf(
        ...                                 x=[x], mean=[0], cov=[[1]])
        >>> sn_pdf2 = lambda x1,x2: multivariate_normal.pdf(
        ...                     x=[x1, x2], mean=[0, 0], cov=[[1, 0], [0, 1]])
        >>> sn1 = CustomDistribution(variables=['x2'], distribution=sn_pdf1)
        >>> sn2 = CustomDistribution(variables=['x1', 'x2'],
        ...                          distribution=sn_pdf2)

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
        Gives the CustomDistribution divide with the other factor.

        Parameters
        ----------
        other: CustomDistribution
            The CustomDistribution to be multiplied.

        Returns
        -------
        CustomDistribution or None:
                        if inplace=True (default) returns None
                        if inplace=False returns a new `CustomDistribution` instance.

        Example
        -------
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> from scipy.stats import multivariate_normal
        >>> sn_pdf1 = lambda x: multivariate_normal.pdf(
        ...                                     x=[x], mean=[0], cov=[[1]])
        >>> sn_pdf2 = lambda x1, x2: multivariate_normal.pdf(
        ...                 x=[x1, x2], mean=[0, 0], cov=[[1, 0], [0, 1]])
        >>> sn1 = CustomDistribution(variables=['x2'], distribution=sn_pdf1)
        >>> sn2 = CustomDistribution(variables=['x1', 'x2'],
        ...                          distribution=sn_pdf2)

        >>> sn3 = sn2.divide(sn1, inplace=False)
        >>> sn3.assignment(0, 0)
        0.3989422804014327

        >>> sn3 = sn2 / sn1
        >>> sn3.assignment(0, 0)
        0.3989422804014327
        """
        if set(other.variables) - set(self.variables):
            raise ValueError("Scope of divisor should be a subset of dividend")

        return self._operate(other, "divide", inplace)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__
