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
        """
        cardinality = np.array(cardinality)
        inhibitor_probability = np.array(inhibitor_probability)
        if len(variables) == len(cardinality) == len(inhibitor_probability) and \
                not inhibitor_probability[inhibitor_probability > 0]:
            self.variables = variables
            self.cardinality = cardinality
            self.inhibitor_probability = inhibitor_probability
        elif inhibitor_probability[inhibitor_probability > 0]:
            raise ValueError("Probability values should be <=1 ")
        elif len(variables) != len(cardinality):
            raise ValueError("Size of variables and cardinality should be same")
        elif len(variables) != len(inhibitor_probability):
            raise ValueError("Size of variables and inhibitor_probability should be same")

