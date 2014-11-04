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
        ...                                     [0.8, 0.8, 0.8, 0.8]],
        ...                        evidence=['diff', 'intel'], evidence_card=[2, 2])
        >>> student.add_cpd([diff_cpd, intel_cpd, grade_cpd])
        >>> model = Inference(student)
        """
        self.variables = model.nodes()
        self.cardinality = {}
        if isinstance(model, BayesianModel):
            self.factors = {}
            for node in model.nodes():
                cpd = model.get_cpd(node)
                cpd_as_factor = cpd.to_factor()
                parents = model.predecessors(node)
                self.cardinality[node] = cpd.variable_card

                # TODO: Add these checks
                # if set(cpd_as_factor.variables) != set([node].extend(parents)):
                #     raise ValueError('The cpd has wrong variables associated with it.')
                # if cpd.marginalize(node) != np.ones(np.prod(cpd.cardinality[1:])):
                #     raise ValueError('The values of the cpd are not correct')

                for var in cpd.variables:
                    try:
                        self.factors[var].append(cpd_as_factor)
                    except KeyError:
                        self.factors[var] = [cpd_as_factor]

        elif isinstance(model, MarkovModel):
            self.factors = {}
            for factor in model.get_factors():

                if factor.variables not in self.variables:
                    raise ValueError('Factors are not consistent with the model')

                for var in factor.variables:
                    try:
                        self.factors[var].append(factor)
                    except KeyError:
                        self.factors[var] = [factor]
