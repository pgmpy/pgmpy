from __future__ import division

from itertools import product
from collections import namedtuple

import numpy as np

from pgmpy.factors.base import BaseFactor
from pgmpy.extern import tabulate
from pgmpy.extern import six
from pgmpy.extern.six.moves import map, range, reduce, zip
from pgmpy.utils import StateNameInit, StateNameDecorator


State = namedtuple('State', ['var', 'state'])


class DiscreteFactor(BaseFactor):
    """
    Base class for DiscreteFactor.

    Public Methods
    --------------
    assignment(index)
    get_cardinality(variable)
    marginalize([variable_list])
    normalize()
    product(*DiscreteFactor)
    reduce([variable_values_list])
    """

    @StateNameInit()
    def __init__(self, variables, cardinality, values):
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
        if isinstance(variables, six.string_types):
            raise TypeError("Variables: Expected type list or array like, got string")

        values = np.array(values, dtype=float)

        if len(cardinality) != len(variables):
            raise ValueError("Number of elements in cardinality must be equal to number of variables")

        if values.size != np.product(cardinality):
            raise ValueError("Values array must be of size: {size}".format(
                size=np.product(cardinality)))

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same")

        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.values = values.reshape(self.cardinality)

    def scope(self):
        """
        Returns the scope of the factor.

        Returns
        -------
        list: List of variable names in the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12))
        >>> phi.scope()
        ['x1', 'x2', 'x3']
        """
        return self.variables

    def get_cardinality(self, variables):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variables: list, array-like
                A list of variable names.

        Returns
        -------
        dict: Dictionary of the form {variable: variable_cardinality}

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_cardinality(['x1'])
        {'x1': 2}
        >>> phi.get_cardinality(['x1', 'x2'])
        {'x1': 2, 'x2': 3}
        """
        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        if not all([var in self.variables for var in variables]):
            raise ValueError("Variable not in scope")

        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    @StateNameDecorator(argument=None, return_val=True)
    def assignment(self, index):
        """
        Returns a list of assignments for the corresponding index.

        Parameters
        ----------
        index: list, array-like
            List of indices whose assignment is to be computed

        Returns
        -------
        list: Returns a list of full assignments of all the variables of the factor.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['diff', 'intel'], [2, 2], np.ones(4))
        >>> phi.assignment([1, 2])
        [[('diff', 0), ('intel', 1)], [('diff', 1), ('intel', 0)]]
        """
        index = np.array(index)

        max_possible_index = np.prod(self.cardinality) - 1
        if not all(i <= max_possible_index for i in index):
            raise IndexError("Index greater than max possible index")

        assignments = np.zeros((len(index), len(self.scope())), dtype=np.int)
        rev_card = self.cardinality[::-1]
        for i, card in enumerate(rev_card):
            assignments[:, i] = index % card
            index = index // card

        assignments = assignments[:, ::-1]

        return [[(key, val) for key, val in zip(self.variables, values)] for values in assignments]

    def identity_factor(self):
        """
        Returns the identity factor.

        Def: The identity factor of a factor has the same scope and cardinality as the original factor,
             but the values for all the assignments is 1. When the identity factor is multiplied with
             the factor it returns the factor itself.

        Returns
        -------
        DiscreteFactor: The identity factor.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi_identity = phi.identity_factor()
        >>> phi_identity.variables
        ['x1', 'x2', 'x3']
        >>> phi_identity.values
        array([[[ 1.,  1.],
                [ 1.,  1.],
                [ 1.,  1.]],

               [[ 1.,  1.],
                [ 1.,  1.],
                [ 1.,  1.]]])
        """
        return DiscreteFactor(self.variables, self.cardinality, np.ones(self.values.size))

    def marginalize(self, variables, inplace=True):
        """
        Modifies the factor with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.marginalize(['x1', 'x3'])
        >>> phi.values
        array([ 14.,  22.,  30.])
        >>> phi.variables
        ['x2']
        """

        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.values = np.sum(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def maximize(self, variables, inplace=True):
        """
        Maximizes the factor with respect to `variables`.

        Parameters
        ----------
        variables: list, array-like
            List of variables with respect to which factor is to be maximized

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [3, 2, 2], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07,
        ...                                              0.00, 0.00, 0.15, 0.21, 0.09, 0.18])
        >>> phi.variables
        ['x1','x2','x3']
        >>> phi.maximize(['x2'])
        >>> phi.variables
        ['x1', 'x3']
        >>> phi.cardinality
        array([3, 2])
        >>> phi.values
        array([[ 0.25,  0.35],
               [ 0.05,  0.07],
               [ 0.15,  0.21]])
        """
        if isinstance(variables, six.string_types):
            raise TypeError("variables: Expected type list or array-like, got type str")

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.values = np.max(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        """
        Normalizes the values of factor so that they sum to 1.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.values
        array([[[ 0,  1],
                [ 2,  3],
                [ 4,  5]],

               [[ 6,  7],
                [ 8,  9],
                [10, 11]]])
        >>> phi.normalize()
        >>> phi.variables
        ['x1', 'x2', 'x3']
        >>> phi.cardinality
        array([2, 3, 2])
        >>> phi.values
        array([[[ 0.        ,  0.01515152],
                [ 0.03030303,  0.04545455],
                [ 0.06060606,  0.07575758]],

               [[ 0.09090909,  0.10606061],
                [ 0.12121212,  0.13636364],
                [ 0.15151515,  0.16666667]]])

        """
        phi = self if inplace else self.copy()

        phi.values = phi.values / phi.values.sum()

        if not inplace:
            return phi

    @StateNameDecorator(argument='values', return_val=None)
    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_state).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce([('x1', 0), ('x2', 0)])
        >>> phi.variables
        ['x3']
        >>> phi.cardinality
        array([2])
        >>> phi.values
        array([0., 1.])
        """
        if isinstance(values, six.string_types):
            raise TypeError("values: Expected type list or array-like, got type str")

        if (any(isinstance(value, six.string_types) for value in values) or
                not all(isinstance(state, (int, np.integer)) for var, state in values)):
            raise TypeError("values: must contain tuples or array-like elements of the form "
                            "(hashable object, type int)")

        phi = self if inplace else self.copy()

        var_index_to_del = []
        slice_ = [slice(None)] * len(self.variables)
        for var, state in values:
            var_index = phi.variables.index(var)
            slice_[var_index] = state
            var_index_to_del.append(var_index)

        var_index_to_keep = sorted(set(range(len(phi.variables))) - set(var_index_to_del))
        # set difference is not gaurenteed to maintain ordering
        phi.variables = [phi.variables[index] for index in var_index_to_keep]
        phi.cardinality = phi.cardinality[var_index_to_keep]
        phi.values = phi.values[tuple(slice_)]

        if not inplace:
            return phi

    def sum(self, phi1, inplace=True):
        """
        DiscreteFactor sum with `phi1`.

        Parameters
        ----------
        phi1: `DiscreteFactor` instance.
            DiscreteFactor to be added.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Example
        -------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi1.sum(phi2, inplace=True)
        >>> phi1.variables
        ['x1', 'x2', 'x3', 'x4']
        >>> phi1.cardinality
        array([2, 3, 2, 2])
        >>> phi1.values
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
                 [55, 77]]]])
        """
        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values += phi1
        else:
            phi1 = phi1.copy()

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[slice_]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[slice_]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], \
                    phi1.variables[axis]
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values + phi1.values

        if not inplace:
            return phi

    def product(self, phi1, inplace=True):
        """
        DiscreteFactor product with `phi1`.

        Parameters
        ----------
        phi1: `DiscreteFactor` instance
            DiscreteFactor to be multiplied.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Example
        -------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi1.product(phi2, inplace=True)
        >>> phi1.variables
        ['x1', 'x2', 'x3', 'x4']
        >>> phi1.cardinality
        array([2, 3, 2, 2])
        >>> phi1.values
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
        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values *= phi1
        else:
            phi1 = phi1.copy()

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi.values = phi.values[slice_]

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

            # modifying phi1 to add new variables
            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))
                phi1.values = phi1.values[slice_]

                phi1.variables.extend(extra_vars)
                # No need to modify cardinality as we don't need it.

            # rearranging the axes of phi1 to match phi
            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], \
                    phi1.variables[axis]
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values * phi1.values

        if not inplace:
            return phi

    def divide(self, phi1, inplace=True):
        """
        DiscreteFactor division by `phi1`.

        Parameters
        ----------
        phi1 : `DiscreteFactor` instance
            The denominator for division.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = DiscreteFactor(['x3', 'x1'], [2, 2], range(1, 5)])
        >>> phi1.divide(phi2)
        >>> phi1.variables
        ['x1', 'x2', 'x3']
        >>> phi1.cardinality
        array([2, 3, 2])
        >>> phi1.values
        array([[[ 0.        ,  0.33333333],
                [ 2.        ,  1.        ],
                [ 4.        ,  1.66666667]],

               [[ 3.        ,  1.75      ],
                [ 4.        ,  2.25      ],
                [ 5.        ,  2.75      ]]])
        """
        phi = self if inplace else self.copy()
        phi1 = phi1.copy()

        if set(phi1.variables) - set(phi.variables):
            raise ValueError("Scope of divisor should be a subset of dividend")

        # Adding extra variables in phi1.
        extra_vars = set(phi.variables) - set(phi1.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.values = phi1.values[slice_]

            phi1.variables.extend(extra_vars)

        # Rearranging the axes of phi1 to match phi
        for axis in range(phi.values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], phi1.variables[axis]
            phi1.values = phi1.values.swapaxes(axis, exchange_index)

        phi.values = phi.values / phi1.values

        # If factor division 0/0 = 0 but is undefined for x/0. In pgmpy we are using
        # np.inf to represent x/0 cases.
        phi.values[np.isnan(phi.values)] = 0

        if not inplace:
            return phi

    def copy(self):
        """
        Returns a copy of the factor.

        Returns
        -------
        DiscreteFactor: copy of the factor

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 3], np.arange(18))
        >>> phi_copy = phi.copy()
        >>> phi_copy.variables
        ['x1', 'x2', 'x3']
        >>> phi_copy.cardinality
        array([2, 3, 3])
        >>> phi_copy.values
        array([[[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8]],

               [[ 9, 10, 11],
                [12, 13, 14],
                [15, 16, 17]]])
        """
        # not creating a new copy of self.values and self.cardinality
        # because __init__ methods does that.
        return DiscreteFactor(self.scope(), self.cardinality, self.values)

    def __str__(self):
        if six.PY2:
            return self._str(phi_or_p='phi', tablefmt="psql")
        else:
            return self._str(phi_or_p='phi')

    def _str(self, phi_or_p="phi", tablefmt="fancy_grid", print_state_names=True):
        """
        Generate the string from `__str__` method.

        Parameters
        ----------
        phi_or_p: 'phi' | 'p'
                'phi': When used for Factors.
                  'p': When used for CPDs.
        print_state_names: boolean
                If True, the user defined state names are displayed.
        """
        string_header = list(map(lambda x: six.text_type(x), self.scope()))
        string_header.append('{phi_or_p}({variables})'.format(phi_or_p=phi_or_p,
                                                              variables=','.join(string_header)))

        value_index = 0
        factor_table = []
        for prob in product(*[range(card) for card in self.cardinality]):
            if self.state_names and print_state_names:
                prob_list = ["{var}({state})".format(
                    var=list(self.variables)[i], state=self.state_names[list(
                        self.variables)[i]][prob[i]])
                             for i in range(len(self.variables))]
            else:
                prob_list = ["{s}_{d}".format(s=list(self.variables)[i], d=prob[i])
                             for i in range(len(self.variables))]

            prob_list.append(self.values.ravel()[value_index])
            factor_table.append(prob_list)
            value_index += 1

        return tabulate(factor_table, headers=string_header, tablefmt=tablefmt, floatfmt=".4f")

    def __repr__(self):
        var_card = ", ".join(['{var}:{card}'.format(var=var, card=card)
                              for var, card in zip(self.variables, self.cardinality)])
        return "<DiscreteFactor representing phi({var_card}) at {address}>".format(address=hex(id(self)), var_card=var_card)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self.sum(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return self.divide(other, inplace=False)

    __div__ = __truediv__

    def __eq__(self, other):
        if not (isinstance(self, DiscreteFactor) and isinstance(other, DiscreteFactor)):
            return False

        elif set(self.scope()) != set(other.scope()):
            return False

        else:
            phi = other.copy()
            for axis in range(self.values.ndim):
                exchange_index = phi.variables.index(self.variables[axis])
                phi.variables[axis], phi.variables[exchange_index] = (phi.variables[exchange_index],
                                                                      phi.variables[axis])
                phi.cardinality[axis], phi.cardinality[exchange_index] = (phi.cardinality[exchange_index],
                                                                          phi.cardinality[axis])
                phi.values = phi.values.swapaxes(axis, exchange_index)

            if phi.values.shape != self.values.shape:
                return False
            elif not np.allclose(phi.values, self.values):
                return False
            elif not all(self.cardinality == phi.cardinality):
                return False
            else:
                return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        sorted_var_hashes = sorted(variable_hashes)
        phi = self.copy()
        for axis in range(phi.values.ndim):
            exchange_index = variable_hashes.index(sorted_var_hashes[axis])
            variable_hashes[axis], variable_hashes[exchange_index] = (variable_hashes[exchange_index],
                                                                      variable_hashes[axis])
            phi.cardinality[axis], phi.cardinality[exchange_index] = (phi.cardinality[exchange_index],
                                                                      phi.cardinality[axis])
            phi.values = phi.values.swapaxes(axis, exchange_index)
        return hash(str(sorted_var_hashes) + str(phi.values) + str(phi.cardinality))


def factor_product(*args):
    """
    Returns factor product over `args`.

    Parameters
    ----------
    args: `DiscreteFactor` instances.
        factors to be multiplied

    Returns
    -------
    DiscreteFactor: `DiscreteFactor` representing factor product over all the `DiscreteFactor` instances in args.

    Examples
    --------
    >>> from pgmpy.factors.discrete import DiscreteFactor, factor_product
    >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
    >>> phi = factor_product(phi1, phi2)
    >>> phi.variables
    ['x1', 'x2', 'x3', 'x4']
    >>> phi.cardinality
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
             [55, 77]]]])
    """
    if not all(isinstance(phi, DiscreteFactor) for phi in args):
        raise TypeError("Arguments must be factors")
    return reduce(lambda phi1, phi2: phi1 * phi2, args)


def factor_divide(phi1, phi2):
    """
    Returns `DiscreteFactor` representing `phi1 / phi2`.

    Parameters
    ----------
    phi1: Factor
        The Dividend.

    phi2: Factor
        The Divisor.

    Returns
    -------
    DiscreteFactor: `DiscreteFactor` representing factor division `phi1 / phi2`.

    Examples
    --------
    >>> from pgmpy.factors.discrete import DiscreteFactor, factor_divide
    >>> phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = DiscreteFactor(['x3', 'x1'], [2, 2], range(1, 5))
    >>> phi = factor_divide(phi1, phi2)
    >>> phi.variables
    ['x1', 'x2', 'x3']
    >>> phi.cardinality
    array([2, 3, 2])
    >>> phi.values
    array([[[ 0.        ,  0.33333333],
            [ 2.        ,  1.        ],
            [ 4.        ,  1.66666667]],

           [[ 3.        ,  1.75      ],
            [ 4.        ,  2.25      ],
            [ 5.        ,  2.75      ]]])
    """
    if not isinstance(phi1, DiscreteFactor) or not isinstance(phi2, DiscreteFactor):
        raise TypeError("phi1 and phi2 should be factors instances")
    return phi1.divide(phi2, inplace=False)
