#!/usr/bin/env python3
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel
from pgmpy.factors import Factor


class Inference:
    """
    Base class for all Inference algorithms.

    Converts BayesianModel and MarkovModel to a uniform representation so that inference
    algorithms can be applied. Also it checks if all the associated CPDs / Factors are
    consistent with the model.
    """
    def __init__(self, model):
        """
        Initialize Inference for a model.

        Parameters
        ----------
        model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
            model for which to initialize the Inference object.

        Examples
        --------
        >>> from pgmpy.Inference import VariableElimination
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2, 0.8]])
        >>> intel_cpd = TabularCPD('intel', 2, [[0.3, 0.7]])
        >>> grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1],
        ...                                     [0.1, 0.1, 0.1, 0.1],
        ...                                     [0.8, 0.8, 0.8, 0.8]])
        >>> student.add_cpd([diff_cpd, intel_cpd, grade_cpd])
        >>> model = Inference(student)
        """
        self.variables = model.nodes()
        if isinstance(model, BayesianModel):
            self.factors = {}
            for node in model.nodes():
                cpd = model.get_cpd(node)
                factor = Factor(cpd.get_variables(), cpd.get_cardinality(), cpd.values)
                for var in cpd.get_variables():
                    try:
                        self.factors[var].append(factor)
                    except KeyError:
                        self.factors[var] = [factor]
                self.factors.append(factor)

        elif isinstance(model, MarkovModel):
            self.factors = {}
            for factor in model.get_factors():
                for var in factor.variables:
                    try:
                        self.factors[var].append(factor)
                    except KeyError:
                        self.factors[var] = [factor]
