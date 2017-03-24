from pgmpy.factors.distributions import BaseDistribution
import numpy as np
from scipy import integrate

class ContinuousDistribution(BaseDistribution):
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
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'],distribution=dirichlet_pdf)
        >>> dirichlet_dist.scope()
        ['x', 'y']
        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError(
                "variables: Expected type: iterable, got: {type}".format(
                    type=type(variables)))

        if len(set(variables)) != len(variables):
            raise ValueError("Multiple variables can't have the same name")

        self.variables = list(variables)
        self._pdf = distribution

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
                    if inplace=False returns a new ContinuousFactor instance.

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
            raise TypeError("variables: Expected type: iterable, "
                            "got: {var_type}".format(var_type=type(values)))

        for var, value in values:
            if var not in self.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        phi = self if inplace else self.copy()

        var_to_remove = [var for var, value in values]
        var_to_keep = [var for var in self.variables if var not in var_to_remove]

        reduced_var_index = [(self.variables.index(var), value) for var, value in values]
        pdf = self._pdf

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
                reduced_args = [arg for arg in reduced_args if arg not in
                                reduced_kwargs.values()]

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
            a new CustomDistribution instance.

        Returns
        -------
        DiscreteFactor or None:
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
        >>> normal_dist.get_scope()
        ['x1', 'x2']
        >>> normal_dist.assignment(1, 1)
        0.058549831524319168
        >>> normal_dist.marginalize(['x2'])
        >>> normal_dist.get_scope()
        ['x1']
        >>> normal_dist.assignment(1)
        0.24197072451914328
        """
        if len(variables) == 0:
            raise ValueError("Shouldn't be calling marginalize over no variable.")

        if not isinstance(variables, (list, tuple, np.ndarray)):
            raise TypeError("variables: Expected type iterable, "
                            "got: {var_type}".format(var_type=type(variables)))

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
        Normalizes the pdf of the continuous distribution so that it
        integrates to 1 over all the variables.

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
        >>> from pgmpy.factors.distributions import CustomDistribution
        >>> from scipy.stats import multivariate_normal
        >>> normal_pdf_x2 = lambda x1, x2: 2 * multivariate_normal.pdf(
        ...                     x=[x1, x2], mean=[0, 0], cov=[[1, 0], [0, 1]])
        >>> normal_dist_x2 = CustomDistribution(variables=['x1', 'x2'],
        ...                                     distribution=normal_pdf_x2)
        >>> normal_dist_x2.assignment(1, 1)
        0.117099663049
        >>> normal_dist = normal_dist_x2.normalize(inplace=False))
        >>> normal_dist.assignment(1, 1)
        0.0585498315243
        """
        phi = self if inplace else self.copy()
        pdf = self._pdf

        pdf_mod = integrate.nquad(pdf, [[-np.inf, np.inf] for var in self.variables])[0]

        phi._pdf = lambda *args: pdf(*args) / pdf_mod

        if not inplace:
            return phi
