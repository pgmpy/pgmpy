#!/usr/bin/env python3

from Exceptions import Exceptions
import numpy as np
from collections import OrderedDict


class Factor:
    """
    Public Methods
    --------------
    assignment(index)
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
            |  x1 |  x2 |  x3 |    phi(x1, x2, x2)|
            -+-----+-----+-----+-------------------+
            | x1_0| x2_0| x3_0|     phi.value(0)  |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_0| x3_0|     phi.value(1)  |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_1| x3_0|     phi.value(2)  |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_1| x3_0|     phi.value(3)  |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_0| x3_1|     phi.value(4)  |
            -+-----+-----+-----+-------------------+
            | x1_1| x2_0| x3_1|     phi.value(5)  |
            -+-----+-----+-----+-------------------+
            | x1_0| x2_1| x3_1|     phi.value(6)  |
             -+-----+-----+-----+-------------------+
            | x1_1| x2_1| x3_1|     phi.value(7)  |
            -+-----+-----+-----+-------------------+
            """
        self.variables = OrderedDict()
        for variable, card in zip(variables, cardinality):
            self.variables[variable] = [variable + '_' + str(index)
                                        for index in range(card)]
        self.cardinality = np.array(cardinality)
        self.value = np.array(value)

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