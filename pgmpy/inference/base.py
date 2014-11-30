#!/usr/bin/env python3

from pgmpy.models import BayesianModel
from pgmpy.models import MarkovModel


class Inference:
    """
    Base class for all inference algorithms.

    Converts BayesianModel and MarkovModel to a uniform representation so that inference
    algorithms can be applied. Also it checks if all the associated CPDs / Factors are
    consistent with the model.
    """
    def __init__(self, model):
        """
        Initialize inference for a model.

        Parameters
        ----------
        model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
            model for which to initialize the inference object.

        Examples
        --------
        >>> from pgmpy.inference import Inference
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2, 0.8]])
        >>> intel_cpd = TabularCPD('intel', 2, [[0.3, 0.7]])
        >>> grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1],
        ...                                     [0.1, 0.1, 0.1, 0.1],
        ...                                     [0.8, 0.8, 0.8, 0.8]],
        ...                        evidence=['diff', 'intel'], evidence_card=[2, 2])
        >>> student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        >>> model = Inference(student)

        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> import numpy as np
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                        ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> factor_a_b = Factor(['Alice', 'Bob'], cardinality=[2, 2], value=np.random.rand(4))
        >>> factor_b_c = Factor(['Bob', 'Charles'], cardinality=[2, 2], value=np.random.rand(4))
        >>> factor_c_d = Factor(['Charles', 'Debbie'], cardinality=[2, 2], value=np.random.rand(4))
        >>> factor_d_a = Factor(['Debbie', 'Alice'], cardinality=[2, 2], value=np.random.rand(4))
        >>> student.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)
        >>> model = Inference(student)
        """
        self.variables = model.nodes()
        self.cardinality = {}
        if isinstance(model, BayesianModel):
            self.factors = {}
            for node in model.nodes():
                cpd = model.get_cpds(node)
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

                if not set(factor.variables.keys()).issubset(set(self.variables)):
                    raise ValueError('Factors are not consistent with the model')

                for index in range(len(factor.variables)):
                    self.cardinality[list(factor.variables.keys())[index]] = factor.cardinality[index]

                for var in factor.variables:
                    try:
                        self.factors[var].append(factor)
                    except KeyError:
                        self.factors[var] = [factor]
