from collections import namedtuple

from pgmpy.factors import BaseDistribution, BaseFactor
from pgmpy.factors.discrete.distributions import CustomDistribution
from pgmpy.utils import StateNameInit, StateNameDecorator


State = namedtuple('State', ['var', 'state'])

class DiscreteFactor(BaseFactor):
    """
    Base class for DiscreteFactor.
    """
    @StateNameInit()
    def __init__(self, variables=None, cardinality=None, values=None, dist=None):
        """
        Initialize a factor class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------+
        |  x1 |  x2 |  x3 |    phi(x1, x2, x3)|
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_0|     phi.value(0)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_1|     phi.value(1)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_0|     phi.value(2)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_1|     phi.value(3)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_0|     phi.value(4)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_1|     phi.value(5)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_0|     phi.value(6)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_1|     phi.value(7)  |
        +-----+-----+-----+-------------------+

        Parameters
        ----------
        variables: list, array-like
            List of variables in the scope of the factor.

        cardinality: list, array_like
            List of cardinalities of each variable. `cardinality` array must have a value
            corresponding to each variable in `variables`.

        values: list, array_like
            List of values of factor.
            A DiscreteFactor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in
            `variables` cycle through their values the fastest.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        >>> phi
        <DiscreteFactor representing phi(x1:2, x2:2, x3:2) at 0x7f8188fcaa90>
        >>> print(phi)
        +------+------+------+-----------------+
        | x1   | x2   | x3   |   phi(x1,x2,x3) |
        |------+------+------+-----------------|
        | x1_0 | x2_0 | x3_0 |          1.0000 |
        | x1_0 | x2_0 | x3_1 |          1.0000 |
        | x1_0 | x2_1 | x3_0 |          1.0000 |
        | x1_0 | x2_1 | x3_1 |          1.0000 |
        | x1_1 | x2_0 | x3_0 |          1.0000 |
        | x1_1 | x2_0 | x3_1 |          1.0000 |
        | x1_1 | x2_1 | x3_0 |          1.0000 |
        | x1_1 | x2_1 | x3_1 |          1.0000 |
        +------+------+------+-----------------+
        """
        # TODO: Deal with the case when dist is not None. Also add conditions.
        self.dist = CustomDistribution(variables=variables,
                                       cardinality=cardinality,
                                       values=values)

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(variables=['x1', 'x2', 'x3'],
        ...                      cardinality=[2, 3, 2],
        ...                      values=np.ones(12))
        >>> phi.scope()
        ['x1', 'x2', 'x3']
        """
        return self.dist.variables

    def get_cardinality(self, variables):
        """
        Returns the cardinality of the variables.

        Parameters
        ----------
        variables: list, iterable
            A list of variable names whose cardinalities would be returned.

        Returns
        -------
        dict: Returns a dict of the form {variable: variable_cardinality}

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(variables=['x1', 'x2', 'x3'],
        ...                      cardinality=[2, 3, 2],
        ...                      values=range(12))
        >>> phi.get_cardinality(['x1'])
        {'x1': 2}
        >>> phi.get_cardinality(['x1', 'x2'])
        {'x1': 2, 'x2': 3}
        """
        return self.dist.get_cardinality(variables=variables)

    def marginalize(self, variables, inplace=True):
        """
        Modifies the distribution with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True, it will modify the factor itself, else would return a
            new factor instance.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None.
                        if inplace=False, returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.marginalize(['x1', 'x3'])
        >>> phi.dist.values
        array([14., 22., 30.])
        >>> phi.dist.variables
        ['x2']
        """
        if inplace:
            self.dist.marginalize(variables)
        else:
            new_dist = self.dist.marginalize(variables, inplace=False)
            return DiscreteFactor(dist=new_dist)

    def normalize(self, inplace=False):
        """
        Normalizes the underlying distribution.

        Parameters
        ----------
        inplace: boolean
            if inplace=True (default), modifies the original factor, else returns a new 
            normalized factor.

        Returns
        -------
        ContinuousFactor or None:
            if inplace=True (default) return None.
            if inplace=False, returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.continuous import DiscreteFactor
        >>> phi = DiscreteFactor(variables=['x1', 'x2', 'x3'],
        ...                      cardinality=[2, 3, 2],
        ...                      values=range(12))
        >>> phi.dist.values
        array([[[ 0,  1],
                [ 2,  3],
                [ 4,  5]],

               [[ 6,  7],
                [ 8,  9],
                [10, 11]]])
        >>> phi.normalize()
        >>> phi.dist.variables
        ['x1', 'x2', 'x3']
        >>> phi.dist.cardinality
        array([2, 3, 2])
        >>> phi.values
        array([[[ 0.        ,  0.01515152],
                [ 0.03030303,  0.04545455],
                [ 0.06060606,  0.07575758]],

               [[ 0.09090909,  0.10606061],
                [ 0.12121212,  0.13636364],
                [ 0.15151515,  0.16666667]]])
        """
        if inplace:
            self.dist.normalize()
        else:
            new_dist = self.dist.normalize(inplace=False)
            return DiscreteFactor(dist=new_dist)

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of the given variable values.

        Parameters
        ----------
        values: list, iterable
                A list of tuples of the form (variable_name, variable_state)

        inplace: boolean
                If inplace=True, it will modify the factor itself, else returns
                a new factor.

        Returns
        -------
        DiscreteFactor or None: If inplace=True (default) returns None,
               if inplace=False returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(variables=['x1', 'x2', 'x3'],
        ...                      cardinality=[2, 3, 2],
        ...                      values=range(12))
        >>> phi.reduce([('x1', 0), ('x2', 0)])
        >>> phi.dist.variables
        ['x3']
        >>> phi.dist.cardinality
        array([2])
        >>> phi.dist.values
        array([0., 1.])
        """
        if inplace:
            self.dist.reduce(values)
        else:
            new_dist = self.dist.reduce(values, inplace=False)
            return DiscreteFactor(dist=new_dist)

    def product(self, phi1, inplace=True):
        """
        DiscreteFactor product with `phi1`.

        Parameters
        ----------
        phi1: `DiscreteFactor` instance.
            DiscreteFactor to be multiplied.

        inplace: boolean
            If inplace=True, modifies the factor itself, else returns a new 
            factor instance.

        Returns
        -------
        DiscreteFactor or None: If inplace=True (default), returns None.
                    If inplace=False, returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete  import DiscreteFactor
        >>> phi1 = DiscreteFactor(variables=['x1', 'x2', 'x3'],
        ...                       cardinality=[2, 3, 2],
        ...                       values=range(12))
        >>> phi2 = DiscreteFactor(variables=['x3', 'x4', 'x1'],
        ...                       cardinality=[2, 2, 2],
        ...                       values=range(8))
        >>> phi1.product(phi2, inplace=True)
        >>> phi1.dist.variables
        ['x1', 'x2', 'x3', 'x4']
        >>> phi1.dist.cardinality
        array([2, 3, 2, 2])
        >>> phi.values
        array([[[[ 0,  0],
                 [ 4,  6]],

                [[ 0,  4],
                 [12, 18]],

                [[ 0,  8],
                 [20, 30]]],


               [[[ 6, 18],
                 [35, 49]],

                [[ 8, 24],
                 [45, 63]],

                [[10, 30],
                 [55, 77]]]]
        """
        if inplace:
            self.dist.product(phi1.dist)
        else:
            new_dist = self.dist.product(phi1.dist, inplace=False)
            return DiscreteFactor(dist=new_dist)

    def copy(self):
        """
        Returns a copy of the factor.

        Returns
        -------
        DiscreteFactor instance: copy of the factor.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(variables=['x1', 'x2', 'x3'],
        ...                      cardinality=[2, 3, 3],
        ...                      values=range(18))
        >>> phi_copy = phi.copy()
        >>> phi_copy.dist.variables
        ['x1', 'x2', 'x3']
        >>> phi_copy.dist.cardinality
        array([2, 3, 3])
        >>> phi_copy.dist.values
        array([[[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8]],

               [[ 9, 10, 11],
                [12, 13, 14],
                [15, 16, 17]]])
        """
        return DiscreteFactor(dist=self.dist.copy())

