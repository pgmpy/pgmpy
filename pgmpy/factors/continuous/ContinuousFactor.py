from pgmpy.factors.base import BaseFactor
from pgmpy.factors import BaseDistribution
from pgmpy.extern import six

class ContinuousFactor(BaseFactor):
    def __init__(self, variables=None, dist=None, **kwargs):
        """
        Initialize a new continuous factor object.

        Parameters
        ----------
        variables: list, iterable (optional)
            The list of varibles on which the factor is defined. Not required if dist
            is an instance of pgmpy.factor.distributions.BaseDistribution
            
            PS: The name of the variables should be same as in the network structure.

        dist: string or instance of pgmpy.factor.distributions.BaseDistribution
            Possible string arguments:
                `gaussian`: If the distribution represented by the factor is a gaussian
                            distribution. It also requires 2 kwargs, mu and sigma.

        **kwargs: Different kwargs are required for different values of `dist`. The
            required arguments are as follows:
            
            If dist='gaussian':
                mean = list, iterable
                    A list of size n (n = no of variables) representing the mean vector
                    of the joint Gausssian Distribution.
                cov = 2-D list, iterable
                    A list of size n x n (n = no of variables) representing the covariance
                    of the joint Gaussian Distribution.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousFactor

        For initializing using a string argument for dist.
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
                                   mean=[1, 2, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        For initializing using an instance of BaseDistribution.
        >>> from pgmpy.factors import GaussianDistribution
        >>> gauss_dist = GaussianDistribution(variables=['A', 'B', 'C'],
                                        mean=[1, 2, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi = ContinuousFactor(dist=gauss_dist)
        """
        if variables:
            if not hasattr(variables, '__iter__') or isinstance(variables, str):
                raise ValueError("variables: Expected type: list/array, Got: {t}".format(t=type(variables)))
            elif not isinstance(dist, six.string_types):
                raise ValueError("dist: Expected type: str, got: {t}".format(t=type(dist)))
            elif dist=='gaussian':
                # import pdb; pdb.set_trace()
                if 'mean' not in kwargs:
                    raise ValueError("If dist='gaussian', the argument `mean` must be passed.")
                elif 'cov' not in kwargs:
                    raise ValueError("If dist='gaussian', the argument `cov` must be passed.")
                else:
                    from pgmpy.factors.continuous.distributions import GaussianDistribution
                    self.dist = GaussianDistribution(variables=variables,
                                                     mean=kwargs['mean'],
                                                     cov=kwargs['cov'])
            else:
                raise ValueError("Please refer to docstring for possible values of dist.")
        else:
            if not isinstance(dist, BaseDistribution):
                raise ValueError("If variables=None, dist must be an instance of BaseDistribution.")
            else:
                self.dist = dist

    def scope(self):
        """
        Return the scope of the factor.

        Returns
        -------
        list: List of variables names in the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
        ...                        mean=[1, 2, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi.scope()
        ['A', 'B', 'C']
        """
        return self.dist.variables

    def marginalize(self, variables, inplace=True):
        """
        Marginalzies the factor over the variables.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True it will modify the factor itself, else returns a new
            factor instance.

        Returns
        -------
        ContinuousFactor or None: 
            If inplace=True (default) returns None.
            If inplace=False, returns a new `ContinuousFactor` instance.                        

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
        ...                        mean=[1, 1, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi.marginalize(variables=['A'], inplace=True)
        >>> phi.scope()
        ['B', 'C']
        >>> phi.dist.mean
        [[1], [1]]
        >>> phi.dist.cov
        [[1, 0], [0, 1]]
        """
        if inplace:
            self.dist.marginalize(variables)
        else:
            new_dist = self.dist.marginalize(variables, inplace=False)
            return ContinuousFactor(dist=new_dist)

    def normalize(self, inplace=True):
        """
        Normalizes the underlying distribution.

        Parameters
        ----------
        inplace: boolean
            If inplace=True (default), modifies the original factor, else returns
            a new normalized factor.

        Returns
        -------
        ContinuousFactor or None: 
            If inplace=True (default) returns None.
            If inplace=False, returns a new `ContinuousFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
                                   mean=[1, 1, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi.normalize()
        """
        if inplace:
            self.dist.normalize()
        else:
            new_dist = self.dist.normalize(inplace=False)
            return ContinuousFactor(dist=new_dist)

    def reduce(self, values, inplace=True):
        """
        Reduce the factor to the context of given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_value).

        inplace: boolean
            If inplace=True, it will modify the factor itself, else returns
            a new factor.

        Returns
        -------
        ContinuousFactor or None:
            If inplace=True (default) returns None.
            If inplace=False, returns a new `ContinuousFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
        ...                        mean=[1, 1, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi.reduce(values=[('A', 1)])
        >>> phi.reduce(values=['A'])
        """
        if inplace:
            self.dist.reduce(values)
        else:
            new_dist = self.dist.reduce(values, inplace=False)
            return ContinuousFactor(dist=new_dist)

    def product(self, other, inplace=True):
        """
        Multiply the factor with other.

        Parameters
        ----------
        other: `ContinuousFactor` instance.
            The factor to be multiplied.

        inplace: boolean
            If inplace=True, modifies the factor itself else returns a new
            `ContinuousFactor` instance.

        Returns
        -------
        ContinuousFactor or None:
                If inplace=True (default), returns None.
                If inplace=False, returns a new `ContinuousFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
        ...                        mean=[1, 1, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi1 = ContinuousFactor(variables=['D'], dist='gaussian',
        ...                         mean=[1], cov=[[1]])
        >>> phi.product(phi1)
        """
        if not type(self.dist) == type(other.dist):
            raise ValueError("other: Expected type: {ex_t}, got: {ot_t}".format(
                ex_t=type(self.dist), ot_t=type(other.dist)))
        if inplace:
            self.dist.product(other)
        else:
            new_dist = self.dist.product(other.dist, inplace=False)
            return ContinuousFactor(dist=new_dist)

    def copy(self):
        """
        Returns a copy of the factor.

        Returns
        -------
        ContinuousFactor: Retuns a copy of the instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> phi = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
                                   mean=[1, 1, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> phi.copy()
        """
        return ContinuousFactor(dist=self.dist.copy())

    def __eq__(self, other):
        return self.dist.__eq__(other.dist)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.__hash__()
