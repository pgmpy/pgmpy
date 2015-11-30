#!/usr/bin/env python3
from itertools import chain

import numpy as np
import networkx as nx

from pgmpy.extern.six.moves import zip
from pgmpy.extern import six


class NoisyOrModel(nx.DiGraph):
    """
    Base class for Noisy-Or models.

    This is an implementation of generalized Noisy-Or models and
    is not limited to Boolean variables and also any arbitrary
    function can be used instead of the boolean OR function.

    Reference: http://xenon.stanford.edu/~srinivas/research/6-UAI93-Srinivas-Generalization-of-Noisy-Or.pdf
    """
    def __init__(self, variables, cardinality, inhibitor_probability):
        # TODO: Accept values of each state so that it could be
        # put into F to compute the final state values of the output
        """
        Init method for NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            array containing names of the variables.

        cardinality: list, tuple, dict (array like)
            array containing integers representing the cardinality
            of the variables.

        inhibitor_probability: list, tuple, dict (array_like)
            array containing the inhibitor probabilities of each variable.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        ...                                                      [0.2, 0.4, 0.7],
        ...                                                      [0.1, 0.4]])
        """
        self.variables = np.array([])
        self.cardinality = np.array([], dtype=np.int)
        self.inhibitor_probability = []
        self.add_variables(variables, cardinality, inhibitor_probability)

    def add_variables(self, variables, cardinality, inhibitor_probability):
        """
        Adds variables to the NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            array containing names of the variables that are to be added.

        cardinality: list, tuple, dict (array like)
            array containing integers representing the cardinality
            of the variables.

        inhibitor_probability: list, tuple, dict (array_like)
            array containing the inhibitor probabilities corresponding to each variable.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        ...                                                      [0.2, 0.4, 0.7],
        ...                                                      [0.1, 0. 4]])
        >>> model.add_variables(['x4'], [3], [0.1, 0.4, 0.2])
        """
        if len(variables) == 1:
            if not isinstance(inhibitor_probability[0], (list, tuple)):
                inhibitor_probability = [inhibitor_probability]

        if len(variables) != len(cardinality):
            raise ValueError("Size of variables and cardinality should be same")
        elif any(cardinal != len(prob_array) for prob_array, cardinal in zip(inhibitor_probability, cardinality)) or \
                len(cardinality) != len(inhibitor_probability):
            raise ValueError("Size of variables and inhibitor_probability should be same")
        elif not all(0 <= item <= 1 for item in chain.from_iterable(inhibitor_probability)):
            raise ValueError("Probability values should be between 0 and 1(both inclusive).")
        else:
            self.variables = np.concatenate((self.variables, variables))
            self.cardinality = np.concatenate((self.cardinality, cardinality))
            self.inhibitor_probability.extend(inhibitor_probability)

    def del_variables(self, variables):
        """
        Deletes variables from the NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            list of variables to be deleted.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        ...                                                      [0.2, 0.4, 0.7],
        ...                                                      [0.1, 0. 4]])
        >>> model.del_variables(['x1'])
        """
        variables = [variables] if isinstance(variables, six.string_types) else set(variables)
        indices = [index for index, variable in enumerate(self.variables) if variable in variables]
        self.variables = np.delete(self.variables, indices, 0)
        self.cardinality = np.delete(self.cardinality, indices, 0)
        self.inhibitor_probability = [prob_array for index, prob_array in enumerate(self.inhibitor_probability)
                                      if index not in indices]

    #
    # def out_prob(self, func):
    #     """
    #     Compute the conditional probability of output variable
    #     given all other variables [P(X|U)] where X is the output
    #     variable and U is the set of input variables.
    #
    #     Parameters
    #     ----------
    #     func: function
    #         The deterministic function which maps input to the
    #         output.
    #
    #     Returns
    #     -------
    #     List of tuples. Each tuple is of the form (state, probability).
    #     """
    #     states = []
    #     from itertools import product
    #     for u in product([(values(var)) for var in self.variables]):
    #         for state in product([(values(var) for var in self.variables)]):
