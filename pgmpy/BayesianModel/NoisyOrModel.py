#!/usr/bin/env python3
import numpy as np


class NoisyOrModel:
    """
    Base class for Noisy-Or Models.

    This is an implementation of generalized Noisy-Or Models and
    is not limited to Boolean variables and also any arbitrary
    function can be used instead of the boolean OR function.

    Reference: http://xenon.stanford.edu/~srinivas/research/6-UAI93-Srinivas-Generalization-of-Noisy-Or.pdf
    """
    def __init__(self, variables, cardinality, inhibitor_probability):
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
        >>> from pgmpy.BayesianModel import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        >>>                                                      [0.2, 0.4, 0.7],
        >>>                                                      [0.1, 0.4]])
        """
        self.variables = []
        self.cardinality = np.array([], dtype=np.int)
        self.inhibitor_probability = np.array([])
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
        >>> from pgmpy.BayesianModel import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        >>>                                                      [0.2, 0.4, 0.7],
        >>>                                                      [0.1, 0. 4]])
        >>> model.add_variables(['x4'], [3], [0.1, 0.4, 0.2])
        """
        cardinality = np.array(cardinality)

        # Converting the inhibitor_probability to a uniform 2D array
        # because else numpy treats it as a 1D array with dtype object.
        inhibitor_probability_list = []
        for prob_array in inhibitor_probability:
            if len(prob_array) < max(cardinality):
                prob_array.extend([0]*(max(cardinality)-len(prob_array)))
                inhibitor_probability_list.append(prob_array)
            else:
                inhibitor_probability_list.append(prob_array)
        inhibitor_probability_uni = np.array(inhibitor_probability_list)

        if inhibitor_probability_uni[inhibitor_probability_uni > 1]:
            raise ValueError("Probability values should be <=1 ")
        elif len(variables) != len(cardinality):
            raise ValueError("Size of variables and cardinality should be same")
        elif (cardinality != [len(prob_array) for prob_array in inhibitor_probability]).any and \
                len(cardinality) != len(inhibitor_probability):
            raise ValueError("Size of variables and inhibitor_probability should be same")
        else:
            self.variables.extend(variables)
            self.cardinality = np.concatenate((self.cardinality, cardinality))
            self.inhibitor_probability = np.concatenate((self.inhibitor_probability, inhibitor_probability_uni))

    def del_variables(self, variables):
        """
        Deletes variables from the NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            list of variables to be deleted.

        Examples
        --------
        >>> from pgmpy.BayesianModel import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        >>>                                                      [0.2, 0.4, 0.7],
        >>>                                                      [0.1, 0. 4]])
        >>> model.delete(['x1'])
        """
        variables = [variables] if isinstance(variables, str) else variables
        for var in variables:
            index = self.variables.index(var)
            self.variables = self.variables.remove(var)
            self.cardinality = np.delete(self.cardinality, index)
            self.inhibitor_probability = np.delete(self.inhibitor_probability, index)
