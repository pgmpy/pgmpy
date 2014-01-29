#!/usr/bin/env python3

from pgmpy import Exceptions
import numpy as np
from collections import OrderedDict


class Factor:
    """
    Public Methods
    --------------
    assignment(index)
    marginalize([variable_list])
    reduce([variable_values_list])
    normalise()
    """

    def __init__(self, variables, cardinality, value):
        """
        Parameters
        ----------
        variables: list
            List of scope of factor
        cardinality: list, array_like
            List of cardinality of each variable
        value: list, array_like
            List or array of values of factor.
            A Factor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in the
            variable field cycle through their values the fastest. More concretely,
            for factor
            >>>phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
            defined above, we have the following mapping from variable
            assignments to the index of the row vector in the value field:
            -+-----+-----+-----+-------------------+
            |  x1 |  x2 |  x3 |    phi(x1, x2, x2) |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_0| x3_0|     phi.value(0)   |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_0| x3_0|     phi.value(1)   |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_1| x3_0|     phi.value(2)   |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_1| x3_0|     phi.value(3)   |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_0| x3_1|     phi.value(4)   |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_0| x3_1|     phi.value(5)   |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_1| x3_1|     phi.value(6)   |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_1| x3_1|     phi.value(7)   |
            -+-----+-----+-----+-------------------+
            """
        self.variables = OrderedDict()
        for variable, card in zip(variables, cardinality):
            self.variables[variable] = [variable + '_' + str(index)
                                        for index in range(card)]
        self.cardinality = np.array(cardinality)
        num_elems = np.cumprod(self.cardinality)[-1]
        self.values = np.array(value, dtype=np.double)
        if not self.values.shape[0] == num_elems:
            raise Exceptions.SizeError("Incompetant value array")

    def assignment(self, index):
        """
        Returns a list of assignments for the corresponding index.
        >>> phi = Factor(['diff', 'intel'], [2, 2])
        >>> phi.assignment([1, 2])
        [['diff_1', 'intel_0'], ['diff_0', 'intel_1']]
        """
        if not isinstance(index, np.ndarray):
            index = np.atleast_1d(index)
        max_index = np.cumprod(self.cardinality)[-1] - 1
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

    def marginalize(self, variables):
        """
        Modifies the factor with marginalized values.
        Paramters
        ---------
        variables: string, list-type
            name of variable to be marginalized
        Example:
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
        Paramters
        ---------
        variable_name: string
            name of variable to be marginalized

        Example:
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi._marginalize_single_variable('x1')
        array([  1.,   5.,   9.,  13.,  17.,  21.])
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
        Reduces the factor to the context of given variable
        values.
        Parameters
        ----------
        values: string, list-type
            name of the variable values

        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce(['x1_0', 'x2_0'])
        >>> phi.values
        array([0, 6])
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
            value_index = int(value_index)
            if not (value_index < self.cardinality[index]):
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
