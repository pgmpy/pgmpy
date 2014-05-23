#!/usr/bin/env python3

from pgmpy import Exceptions
import numpy as np
from collections import OrderedDict
from pgmpy.Factor._factor_product import _factor_product
import functools


class Factor:
    """
    Base class for *Factor*.

    Public Methods
    --------------
    assignment(index)
    get_cardinality(variable)
    marginalize([variable_list])
    normalise()
    product(*factors)
    reduce([variable_values_list])
    """

    def __init__(self, variables, cardinality, value):
        """
        Initialize a Factor class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------+
        |  x1 |  x2 |  x3 |    phi(x1, x2, x2)|
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_0|     phi.value(0)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_0|     phi.value(1)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_0|     phi.value(2)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_0|     phi.value(3)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_1|     phi.value(4)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_1|     phi.value(5)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_1|     phi.value(6)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_1|     phi.value(7)  |
        +-----+-----+-----+-------------------+

        Parameters
        ----------
        variables: list
            List of scope of factor
        cardinality: list, array_like
            List of cardinality of each variable
        value: list, array_like
            List or array of values of factor.
            A Factor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in
            the variable field cycle through their values the fastest. More
            concretely, for factor

        Examples
        --------
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        """
        self.variables = OrderedDict()
        for variable, card in zip(variables, cardinality):
            self.variables[variable] = [variable + '_' + str(index)
                                        for index in range(card)]
        self.cardinality = np.array(cardinality)
        self.values = np.array(value, dtype=np.double)
        if not self.values.shape[0] == np.prod(self.cardinality):
            raise Exceptions.SizeError("Incompetant value array")

    def assignment(self, index):
        """
        Returns a list of assignments for the corresponding index.

        Parameters
        ----------
        index: integer, list-type, ndarray
            index or indices whose assignment is to be computed

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> import numpy as np
        >>> phi = Factor(['diff', 'intel'], [2, 2], np.ones(4))
        >>> phi.assignment([1, 2])
        [['diff_1', 'intel_0'], ['diff_0', 'intel_1']]
        """
        if not isinstance(index, np.ndarray):
            index = np.atleast_1d(index)
        max_index = np.prod(self.cardinality) - 1
        if not all(i <= max_index for i in index):
            raise IndexError("Index greater than max possible index")
        mat = np.floor(np.tile(np.atleast_2d(index).T,
                               (1, self.cardinality.shape[0])) /
                       np.tile(np.cumprod(
                           np.concatenate(([1], self.cardinality[:-1]))),
                               (index.shape[0], 1)))\
              % np.tile(self.cardinality, (index.shape[0], 1))
        mat = mat.astype('int')
        return [[self.variables[key][val] for key, val in
                 zip(self.variables.keys(), values)] for values in mat]

    def get_cardinality(self, variable):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variable: string

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_cardinality('x1')
        2
        """
        if variable not in self.variables:
            raise Exceptions.ScopeError("%s not in scope" % variable)
        return self.cardinality[list(self.variables.keys()).index(variable)]

    def marginalize(self, variables):
        """
        Modifies the factor with marginalized values.

        Paramters
        ---------
        variables: string, list-type
            name of variable to be marginalized

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.marginalize(['x1', 'x3'])
        >>> phi.values
        array([ 14.,  22.,  30.])
        >>> phi.variables
        OrderedDict([('x2', ['x2_0', 'x2_1', 'x2_2'])])
        """
        if not isinstance(variables, list):
            variables = [variables]
        for variable in variables:
            if variable not in self.variables:
                raise Exceptions.ScopeError("%s not in scope" % variable)
        for variable in variables:
            self.values = self._marginalize_single_variable(variable)
            index = list(self.variables.keys()).index(variable)
            del(self.variables[variable])
            self.cardinality = np.delete(self.cardinality, index)

    def _marginalize_single_variable(self, variable):
        """
        Returns marginalised factor for a single variable

        Parameters
        ---------
        variable_name: string
            name of variable to be marginalized

        """
        index = list(self.variables.keys()).index(variable)
        cum_cardinality = np.concatenate(([1], np.cumprod(self.cardinality)))
        num_elements = cum_cardinality[-1]
        sum_index = [j for i in range(0, num_elements,
                                      cum_cardinality[index+1])
                     for j in range(i, i+cum_cardinality[index])]
        marg_factor = np.zeros(num_elements/self.cardinality[index])
        for i in range(self.cardinality[index]):
            marg_factor += self.values[np.array(sum_index) +
                                       i*cum_cardinality[index]]
        return marg_factor

    def normalize(self):
        """
        Normalizes the values of factor so that they sum to 1.
        """
        self.values = self.values/np.sum(self.values)

    def reduce(self, values):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: string, list-type
            name of the variable values

        Examples
        --------
        >>> from pgmpy.Factor import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce(['x1_0', 'x2_0'])
        >>> phi.values
        array([0., 6.])
        """
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if not '_' in value:
                raise TypeError("Values should be in the form of "
                                "variablename_index")
            var, value_index = value.split('_')
            if not var in self.variables:
                raise Exceptions.ScopeError("%s not in scope" % var)
            index = list(self.variables.keys()).index(var)
            if not (int(value_index) < self.cardinality[index]):
                raise Exceptions.SizeError("Value is "
                                           "greater than max possible value")
            cum_cardinality = np.concatenate(([1],
                                              np.cumprod(self.cardinality)))
            num_elements = cum_cardinality[-1]
            index_arr = [j for i in range(0, num_elements,
                                          cum_cardinality[index+1])
                         for j in range(i, i+cum_cardinality[index])]
            self.values = self.values[np.array(index_arr)]
            del(self.variables[var])
            self.cardinality = np.delete(self.cardinality, index)

    def product(self, *factors):
        """
        Returns the factor product with factors.

        Parameters
        ----------
        *factors: Factor1, Factor2, ...
            Factors to be multiplied

        Example
        -------
        >>> from pgmpy.Factor import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi = phi1.product(phi2)
        >>> phi.variables
        OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']),
                ('x3', ['x3_0', 'x3_1']), ('x4', ['x4_0', 'x4_1'])])
        """
        return factor_product(self, *factors)

    def __str__(self):
        return self._str('phi')

    def _str(self, phi_or_p):
        string = ""
        for var in self.variables:
            string += str(var) + "\t"
        string += phi_or_p + '(' + ', '.join(self.variables) + ')'
        string += "\n"

        #fun and gen are functions to generate the different values of variables in the table.
        #gen starts with giving fun initial value of b=[0, 0, 0] then fun tries to increment it
        #by 1.
        def fun(b, index=len(self.cardinality)-1):
            b[index] += 1
            if b[index] == self.cardinality[index]:
                b[index] = 0
                fun(b, index-1)
            return b

        def gen():
            b = [0] * len(self.variables)
            yield b
            for i in range(np.prod(self.cardinality)-1):
                yield fun(b)

        value_index = 0
        for prob in gen():
            prob_list = [list(self.variables)[i] + '_' + str(prob[i]) for i in range(len(self.variables))]
            string += '\t'.join(prob_list) + '\t' + str(self.values[value_index]) + '\n'
            value_index += 1

        return string

    def __mul__(self, other):
        return self.product(other)

    def __eq__(self, other):
        if self.variables == other.variables and self.cardinality == other.cardinality and self.values == other.values:
            return True
        else:
            return False


def _bivar_factor_product(phi1, phi2):
    """
    Returns product of two factors.

    Parameters
    ----------
    phi1: Factor

    phi2: Factor

    See Also
    --------
    factor_product
    """
    vars1 = list(phi1.variables.keys())
    vars2 = list(phi2.variables.keys())
    common_var_list = [var1 for var1 in vars1 for var2 in vars2
                       if var1 == var2]
    if common_var_list:
        common_var_index_list = np.array([[vars1.index(var), vars2.index(var)]
                                          for var in common_var_list])
        common_card_product = np.prod([phi1.cardinality[index[0]] for index
                                       in common_var_index_list])
        size = np.prod(phi1.cardinality) * np.prod(
            phi2.cardinality) / common_card_product
        product = _factor_product(phi1.values,
                                  phi2.values,
                                  size,
                                  common_var_index_list,
                                  phi1.cardinality,
                                  phi2.cardinality)
        variables = vars1
        variables.extend(var for var in phi2.variables
                         if var not in common_var_list)
        cardinality = list(phi1.cardinality)
        cardinality.extend(phi2.get_cardinality(var) for var in phi2.variables
                           if var not in common_var_list)
        phi = Factor(variables, cardinality, product)
        return phi
    else:
        size = np.prod(phi1.cardinality) * np.prod(phi2.cardinality)
        product = _factor_product(phi1.values,
                                  phi2.values,
                                  size)
        variables = vars1 + vars2
        cardinality = list(phi1.cardinality) + list(phi2.cardinality)
        phi = Factor(variables, cardinality, product)
        return phi


def factor_product(*args):
    """
    Returns factor product of multiple factors.

    Parameters
    ----------
    factor1, factor2, .....: Factor
        factors to be multiplied

    Examples
    --------
    >>> from pgmpy.Factor import Factor, factor_product
    >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
    >>> phi = factor_product(phi1, phi2)
    >>> phi.variables
    OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']),
                ('x3', ['x3_0', 'x3_1']), ('x4', ['x4_0', 'x4_1'])])
    """
    if not all(isinstance(phi, Factor) for phi in args):
        raise TypeError("Input parameters must be factors")
    return functools.reduce(_bivar_factor_product, args)